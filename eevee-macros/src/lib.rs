use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    FnArg, ItemFn, ReturnType, Token, Type,
};

struct TypeParam {
    name: syn::Ident,
    variants: Vec<syn::Type>,
}

impl Parse for TypeParam {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![:]>()?;

        let mut variants = Vec::new();
        variants.push(input.parse::<Type>()?);

        while input.peek(Token![|]) {
            input.parse::<Token![|]>()?;
            variants.push(input.parse::<Type>()?);
        }

        Ok(TypeParam { name, variants })
    }
}

struct FnMatrixInput {
    type_params: Vec<TypeParam>,
    functions: Vec<ItemFn>,
    blocks: Vec<syn::Block>,
}

impl Parse for FnMatrixInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut type_params = Vec::new();
        let mut functions = Vec::new();
        let mut blocks = Vec::new();

        // Parse type parameters (A: T | T, B: T | T, etc.)
        loop {
            let checkpoint = input.fork();
            match checkpoint.parse::<TypeParam>() {
                Ok(param) => {
                    input.parse::<TypeParam>()?;
                    type_params.push(param);
                    let _ = input.parse::<Token![,]>();
                }
                Err(_) => break,
            }
        }

        // Parse function definitions or bare blocks
        while !input.is_empty() {
            if input.peek(syn::token::Brace) {
                blocks.push(input.parse()?);
            } else {
                functions.push(input.parse()?);
            }
        }

        Ok(FnMatrixInput {
            type_params,
            functions,
            blocks,
        })
    }
}

/// Generate multiple function variants or blocks by substituting types.
///
/// **Function mode** — generates one monomorphised function per type combination:
/// ```ignore
/// fn_matrix! {
///     A: Type1 | Type2,
///     B: Type3 | Type4,
///
///     pub fn foo(x: A) -> B {
///         B::from(x)
///     }
/// }
/// ```
///
/// **Block mode** — emits one scoped block per type combination with types
/// substituted and sentinels replaced:
/// - `PERM_ID` → `"A_B"` (underscore-separated)
/// - `BENCH_ID` → `"A/B"` (slash-separated)
/// - `const FN_MATRIX_NAME: &'static str = "A_B"` injected at block start
///
/// ```ignore
/// fn_matrix! {
///     A: Type1 | Type2,
///     B: Type3 | Type4,
///     {
///         let name = PERM_ID;
///         process::<A, B>(name);
///     }
/// }
/// ```
#[proc_macro]
pub fn fn_matrix(input: TokenStream) -> TokenStream {
    let parsed: FnMatrixInput = match syn::parse(input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };

    if parsed.type_params.is_empty() {
        return syn::Error::new_spanned(
            proc_macro2::TokenStream::new(),
            "fn_matrix requires at least one type parameter",
        )
        .to_compile_error()
        .into();
    }

    // Generate all combinations via Cartesian product
    let mut combinations = Vec::new();
    generate_combinations(&parsed.type_params, 0, Vec::new(), &mut combinations);

    let mut output = quote! {};

    for combo in &combinations {
        // Build a map of param name -> concrete type for this combo
        let type_map: std::collections::HashMap<String, &syn::Type> = parsed
            .type_params
            .iter()
            .zip(combo.iter())
            .map(|(p, t)| (p.name.to_string(), *t))
            .collect();

        for func in &parsed.functions {
            output.extend(generate_function_variant(
                func,
                &parsed.type_params,
                combo,
                &type_map,
            ));
        }

        for block in &parsed.blocks {
            output.extend(generate_block_variant(
                block,
                &parsed.type_params,
                combo,
                &type_map,
            ));
        }
    }

    TokenStream::from(output)
}

// ---------------------------------------------------------------------------
// Cartesian product generation
// ---------------------------------------------------------------------------

fn generate_combinations<'a>(
    params: &'a [TypeParam],
    index: usize,
    current: Vec<&'a syn::Type>,
    results: &mut Vec<Vec<&'a syn::Type>>,
) {
    if index == params.len() {
        results.push(current);
        return;
    }
    for variant in &params[index].variants {
        let mut next = current.clone();
        next.push(variant);
        generate_combinations(params, index + 1, next, results);
    }
}

// ---------------------------------------------------------------------------
// Name helpers
// ---------------------------------------------------------------------------

fn type_last_ident(t: &syn::Type) -> String {
    match t {
        syn::Type::Path(p) => p
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default(),
        _ => "Unknown".to_string(),
    }
}

fn perm_parts(_params: &[TypeParam], combo: &[&syn::Type]) -> Vec<String> {
    combo.iter().map(|t| type_last_ident(t)).collect()
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

fn generate_function_variant(
    func: &ItemFn,
    params: &[TypeParam],
    combo: &[&syn::Type],
    type_map: &std::collections::HashMap<String, &syn::Type>,
) -> proc_macro2::TokenStream {
    let mut new_fn = func.clone();

    let orig_name = &func.sig.ident;
    let mut name_parts = vec![orig_name.to_string()];
    name_parts.extend(perm_parts(params, combo));
    let new_name = syn::Ident::new(&name_parts.join("_").to_lowercase(), orig_name.span());
    new_fn.sig.ident = new_name;

    replace_types_in_fn(&mut new_fn, type_map, None, None);

    quote! { #new_fn }
}

fn generate_block_variant(
    block: &syn::Block,
    params: &[TypeParam],
    combo: &[&syn::Type],
    type_map: &std::collections::HashMap<String, &syn::Type>,
) -> proc_macro2::TokenStream {
    let parts = perm_parts(params, combo);
    let perm_id = parts.join("_");
    let bench_id = parts.join("/");

    let mut new_block = block.clone();
    replace_types_in_block(&mut new_block, type_map, Some(&perm_id), Some(&bench_id));

    // Inject `const FN_MATRIX_NAME` at the top of the block
    let name_lit = syn::LitStr::new(&perm_id, proc_macro2::Span::call_site());
    let const_stmt: syn::Stmt = syn::parse2(quote! {
        const FN_MATRIX_NAME: &'static str = #name_lit;
    })
    .expect("const stmt is always valid");
    new_block.stmts.insert(0, const_stmt);

    quote! { #new_block }
}

// ---------------------------------------------------------------------------
// Type substitution
// ---------------------------------------------------------------------------

fn replace_types_in_fn(
    func: &mut ItemFn,
    type_map: &std::collections::HashMap<String, &syn::Type>,
    perm_id: Option<&str>,
    bench_id: Option<&str>,
) {
    for arg in &mut func.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            replace_type_in_type(&mut pat_type.ty, type_map);
        }
    }
    if let ReturnType::Type(_, ty) = &mut func.sig.output {
        replace_type_in_type(ty, type_map);
    }
    replace_types_in_block(&mut func.block, type_map, perm_id, bench_id);
}

/// Substitute type params in a type position, recursing into generic arguments.
fn replace_type_in_type(
    ty: &mut Box<syn::Type>,
    type_map: &std::collections::HashMap<String, &syn::Type>,
) {
    if let syn::Type::Path(p) = ty.as_mut() {
        // Single-segment bare ident — direct substitution
        if p.path.segments.len() == 1 {
            let seg = &p.path.segments[0];
            if matches!(seg.arguments, syn::PathArguments::None) {
                if let Some(substitution) = type_map.get(seg.ident.to_string().as_str()) {
                    **ty = (*substitution).clone();
                    return;
                }
            }
        }
        // Recurse into generic arguments (e.g. Vec<A>, Trait<B, C>)
        for seg in &mut p.path.segments {
            if let syn::PathArguments::AngleBracketed(args) = &mut seg.arguments {
                for arg in &mut args.args {
                    if let syn::GenericArgument::Type(inner) = arg {
                        let mut boxed = Box::new(inner.clone());
                        replace_type_in_type(&mut boxed, type_map);
                        *inner = *boxed;
                    }
                }
            }
        }
    }
}

fn replace_types_in_block(
    block: &mut syn::Block,
    type_map: &std::collections::HashMap<String, &syn::Type>,
    perm_id: Option<&str>,
    bench_id: Option<&str>,
) {
    for stmt in &mut block.stmts {
        replace_types_in_stmt(stmt, type_map, perm_id, bench_id);
    }
}

fn replace_types_in_stmt(
    stmt: &mut syn::Stmt,
    type_map: &std::collections::HashMap<String, &syn::Type>,
    perm_id: Option<&str>,
    bench_id: Option<&str>,
) {
    match stmt {
        syn::Stmt::Local(local) => {
            // Substitute in the declared type annotation (e.g. `let x: Vec<A> = ...`)
            if let syn::Pat::Type(pat_type) = &mut local.pat {
                replace_type_in_type(&mut pat_type.ty, type_map);
            }
            if let Some(init) = &mut local.init {
                replace_types_in_expr(&mut init.expr, type_map, perm_id, bench_id);
            }
        }
        syn::Stmt::Item(_) => {}
        syn::Stmt::Expr(expr, _) => {
            replace_types_in_expr(expr, type_map, perm_id, bench_id);
        }
        syn::Stmt::Macro(_) => {}
    }
}

fn replace_types_in_expr(
    expr: &mut syn::Expr,
    type_map: &std::collections::HashMap<String, &syn::Type>,
    perm_id: Option<&str>,
    bench_id: Option<&str>,
) {
    match expr {
        syn::Expr::Path(p) => {
            // Substitute QSelf type in `<A as Trait<B>>::method`
            if let Some(qself) = &mut p.qself {
                replace_type_in_type(&mut qself.ty, type_map);
            }
            // Substitute types in generic args of path segments
            for seg in &mut p.path.segments {
                if let syn::PathArguments::AngleBracketed(args) = &mut seg.arguments {
                    for arg in &mut args.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            let mut boxed = Box::new(inner.clone());
                            replace_type_in_type(&mut boxed, type_map);
                            *inner = *boxed;
                        }
                    }
                }
            }
            // Sentinel and type param replacements (only for simple paths without qself)
            if p.qself.is_none() {
                if let Some(first_segment) = p.path.segments.first() {
                    let ident_str = first_segment.ident.to_string();
                    if ident_str == "PERM_ID" && p.path.segments.len() == 1 {
                        if let Some(s) = perm_id {
                            *expr = syn::parse2(quote! { #s }).unwrap_or_else(|_| expr.clone());
                            return;
                        }
                    }
                    if ident_str == "BENCH_ID" && p.path.segments.len() == 1 {
                        if let Some(s) = bench_id {
                            *expr = syn::parse2(quote! { #s }).unwrap_or_else(|_| expr.clone());
                            return;
                        }
                    }
                    // Type param substitutions
                    if let Some(substitution) = type_map.get(&ident_str) {
                        if p.path.segments.len() == 1 {
                            *expr = syn::parse2(quote! { #substitution })
                                .unwrap_or_else(|_| expr.clone());
                        } else {
                            // Handle X::method by prepending X and appending remaining segments
                            let remaining =
                                p.path.segments.iter().skip(1).cloned().collect::<Vec<_>>();
                            let mut new_path = match substitution {
                                syn::Type::Path(path_type) => path_type.path.clone(),
                                _ => syn::parse2::<syn::Path>(quote! { #substitution }).unwrap(),
                            };
                            for seg in remaining {
                                new_path.segments.push(seg);
                            }
                            p.path = new_path;
                        }
                    }
                }
            }
        }
        syn::Expr::Call(call) => {
            replace_types_in_expr(&mut call.func, type_map, perm_id, bench_id);
            for arg in &mut call.args {
                replace_types_in_expr(arg, type_map, perm_id, bench_id);
            }
        }
        syn::Expr::MethodCall(mc) => {
            replace_types_in_expr(&mut mc.receiver, type_map, perm_id, bench_id);
            // Turbofish type args on method calls (e.g. .collect::<Vec<A>>())
            if let Some(turbofish) = &mut mc.turbofish {
                for arg in &mut turbofish.args {
                    if let syn::GenericArgument::Type(inner) = arg {
                        let mut boxed = Box::new(inner.clone());
                        replace_type_in_type(&mut boxed, type_map);
                        *inner = *boxed;
                    }
                }
            }
            for arg in &mut mc.args {
                replace_types_in_expr(arg, type_map, perm_id, bench_id);
            }
        }
        syn::Expr::Block(b) => {
            replace_types_in_block(&mut b.block, type_map, perm_id, bench_id);
        }
        syn::Expr::Closure(cl) => {
            if let ReturnType::Type(_, ty) = &mut cl.output {
                replace_type_in_type(ty, type_map);
            }
            replace_types_in_expr(&mut cl.body, type_map, perm_id, bench_id);
        }
        syn::Expr::If(e) => {
            replace_types_in_expr(&mut e.cond, type_map, perm_id, bench_id);
            replace_types_in_block(&mut e.then_branch, type_map, perm_id, bench_id);
            if let Some((_, else_branch)) = &mut e.else_branch {
                replace_types_in_expr(else_branch, type_map, perm_id, bench_id);
            }
        }
        syn::Expr::Match(m) => {
            replace_types_in_expr(&mut m.expr, type_map, perm_id, bench_id);
            for arm in &mut m.arms {
                replace_types_in_expr(&mut arm.body, type_map, perm_id, bench_id);
            }
        }
        syn::Expr::Return(r) => {
            if let Some(val) = &mut r.expr {
                replace_types_in_expr(val, type_map, perm_id, bench_id);
            }
        }
        syn::Expr::Tuple(t) => {
            for elem in &mut t.elems {
                replace_types_in_expr(elem, type_map, perm_id, bench_id);
            }
        }
        syn::Expr::Reference(r) => {
            replace_types_in_expr(&mut r.expr, type_map, perm_id, bench_id);
        }
        syn::Expr::Unary(u) => {
            replace_types_in_expr(&mut u.expr, type_map, perm_id, bench_id);
        }
        syn::Expr::Paren(p) => {
            replace_types_in_expr(&mut p.expr, type_map, perm_id, bench_id);
        }
        _ => {}
    }
}
