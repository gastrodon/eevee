use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    FnArg, ItemFn, ReturnType, Token,
};

struct TypeParam {
    name: syn::Ident,
    variants: Vec<proc_macro2::TokenStream>,
}

impl Parse for TypeParam {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![:]>()?;

        let mut variants = Vec::new();

        loop {
            // Always consume tokens until | or , to support forward references like Recurrent<C>
            // This is more flexible and handles both simple types and complex generic types
            let mut tokens = proc_macro2::TokenStream::new();
            let mut depth = 0;

            while !input.is_empty() {
                if depth == 0 && (input.peek(Token![|]) || input.peek(Token![,])) {
                    break;
                }

                let token: proc_macro2::TokenTree = input.parse()?;

                // Track angle bracket and paren depth for generics and nested constructs
                match &token {
                    proc_macro2::TokenTree::Punct(p) => {
                        let ch = p.as_char();
                        if ch == '<' {
                            depth += 1;
                        } else if ch == '>' {
                            if depth > 0 {
                                depth -= 1;
                            }
                        }
                    }
                    _ => {}
                }

                tokens.extend(std::iter::once(token));
            }

            variants.push(tokens);

            if input.peek(Token![|]) {
                input.parse::<Token![|]>()?;
            } else {
                break;
            }
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
        // Build a map of param name -> concrete type tokens for this combo
        // First pass: build initial map
        let mut type_map: std::collections::HashMap<String, proc_macro2::TokenStream> = parsed
            .type_params
            .iter()
            .zip(combo.iter())
            .map(|(p, t)| (p.name.to_string(), t.clone()))
            .collect();

        // Second pass: recursively substitute type parameters within other type parameters
        // (e.g., substitute C in "Recurrent<C>")
        let original_map = type_map.clone();
        for type_param in &parsed.type_params {
            let param_name = type_param.name.to_string();
            if let Some(tokens) = type_map.get_mut(&param_name) {
                let mut substituted = tokens.clone();
                for other_param in &parsed.type_params {
                    let other_name = other_param.name.to_string();
                    if other_name != param_name {
                        if let Some(replacement) = original_map.get(&other_name) {
                            // Simple token-level substitution for identifiers
                            let mut new_tokens = proc_macro2::TokenStream::new();
                            let mut found = false;

                            for token in substituted.clone() {
                                if let proc_macro2::TokenTree::Ident(ident) = &token {
                                    if ident.to_string() == other_name {
                                        new_tokens.extend(replacement.clone());
                                        found = true;
                                        continue;
                                    }
                                }
                                new_tokens.extend(std::iter::once(token));
                            }

                            if found {
                                substituted = new_tokens;
                            }
                        }
                    }
                }
                *tokens = substituted;
            }
        }

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
    current: Vec<proc_macro2::TokenStream>,
    results: &mut Vec<Vec<proc_macro2::TokenStream>>,
) {
    if index == params.len() {
        results.push(current);
        return;
    }
    for variant in &params[index].variants {
        let mut next = current.clone();
        next.push(variant.clone());
        generate_combinations(params, index + 1, next, results);
    }
}

// ---------------------------------------------------------------------------
// Name helpers
// ---------------------------------------------------------------------------

fn extract_idents_from_tokens(tokens: &proc_macro2::TokenStream) -> String {
    let mut result = Vec::new();

    for token in tokens.clone() {
        match token {
            proc_macro2::TokenTree::Ident(ident) => {
                result.push(ident.to_string());
            }
            proc_macro2::TokenTree::Punct(p) if p.as_char() == '<' => {
                break; // Stop before generics
            }
            _ => {}
        }
    }

    result.join("").to_lowercase()
}

fn perm_parts(_params: &[TypeParam], combo: &[proc_macro2::TokenStream]) -> Vec<String> {
    combo
        .iter()
        .map(|t| extract_idents_from_tokens(t))
        .collect()
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

fn generate_function_variant(
    func: &ItemFn,
    params: &[TypeParam],
    combo: &[proc_macro2::TokenStream],
    type_map: &std::collections::HashMap<String, proc_macro2::TokenStream>,
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
    combo: &[proc_macro2::TokenStream],
    type_map: &std::collections::HashMap<String, proc_macro2::TokenStream>,
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

/// Substitute type param identifiers in a token stream (e.g. inside macros)
fn substitute_in_token_stream(
    tokens: &proc_macro2::TokenStream,
    type_map: &std::collections::HashMap<String, proc_macro2::TokenStream>,
) -> proc_macro2::TokenStream {
    let mut out = proc_macro2::TokenStream::new();
    for tt in tokens.clone() {
        match tt {
            proc_macro2::TokenTree::Ident(id) => {
                if let Some(repl) = type_map.get(id.to_string().as_str()) {
                    out.extend(repl.clone());
                } else {
                    out.extend(std::iter::once(proc_macro2::TokenTree::Ident(id)));
                }
            }
            proc_macro2::TokenTree::Group(g) => {
                let inner = substitute_in_token_stream(&g.stream(), type_map);
                out.extend(std::iter::once(proc_macro2::TokenTree::Group(
                    proc_macro2::Group::new(g.delimiter(), inner),
                )));
            }
            _ => out.extend(std::iter::once(tt)),
        }
    }
    out
}

fn replace_types_in_fn(
    func: &mut ItemFn,
    type_map: &std::collections::HashMap<String, proc_macro2::TokenStream>,
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
    type_map: &std::collections::HashMap<String, proc_macro2::TokenStream>,
) {
    if let syn::Type::Path(p) = ty.as_mut() {
        // Single-segment bare ident — direct substitution
        if p.path.segments.len() == 1 {
            let seg = &p.path.segments[0];
            if matches!(seg.arguments, syn::PathArguments::None) {
                if let Some(substitution) = type_map.get(seg.ident.to_string().as_str()) {
                    // Parse the TokenStream back into a Type
                    if let Ok(new_type) = syn::parse2::<syn::Type>(substitution.clone()) {
                        **ty = new_type;
                        return;
                    }
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
    type_map: &std::collections::HashMap<String, proc_macro2::TokenStream>,
    perm_id: Option<&str>,
    bench_id: Option<&str>,
) {
    for stmt in &mut block.stmts {
        replace_types_in_stmt(stmt, type_map, perm_id, bench_id);
    }
}

fn replace_types_in_stmt(
    stmt: &mut syn::Stmt,
    type_map: &std::collections::HashMap<String, proc_macro2::TokenStream>,
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
        syn::Stmt::Macro(m) => {
            m.mac.tokens = substitute_in_token_stream(&m.mac.tokens, type_map);
        }
    }
}

fn replace_types_in_expr(
    expr: &mut syn::Expr,
    type_map: &std::collections::HashMap<String, proc_macro2::TokenStream>,
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
                            *expr =
                                syn::parse2(substitution.clone()).unwrap_or_else(|_| expr.clone());
                        } else {
                            // Handle X::method by prepending X and appending remaining segments
                            let remaining =
                                p.path.segments.iter().skip(1).cloned().collect::<Vec<_>>();
                            let new_path = syn::parse2::<syn::Path>(substitution.clone())
                                .unwrap_or_else(|_| syn::Path {
                                    leading_colon: None,
                                    segments: Default::default(),
                                });
                            let mut final_path = new_path;
                            for seg in remaining {
                                final_path.segments.push(seg);
                            }
                            p.path = final_path;
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
            // Handle closure parameter types (e.g. |x: A| where A is a type param)
            for input in &mut cl.inputs {
                if let syn::Pat::Type(pat_type) = input {
                    replace_type_in_type(&mut pat_type.ty, type_map);
                }
            }
            // Handle return type
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
        syn::Expr::Macro(m) => {
            m.mac.tokens = substitute_in_token_stream(&m.mac.tokens, type_map);
        }
        syn::Expr::Array(a) => {
            for elem in &mut a.elems {
                replace_types_in_expr(elem, type_map, perm_id, bench_id);
            }
        }
        syn::Expr::Struct(s) => {
            // Substitute in struct path generics (e.g., Struct::<A>::new())
            if let syn::PathArguments::AngleBracketed(args) = &mut s.path.segments.last_mut().unwrap().arguments {
                for arg in &mut args.args {
                    if let syn::GenericArgument::Type(inner) = arg {
                        let mut boxed = Box::new(inner.clone());
                        replace_type_in_type(&mut boxed, type_map);
                        *inner = *boxed;
                    }
                }
            }
            for field in &mut s.fields {
                replace_types_in_expr(&mut field.expr, type_map, perm_id, bench_id);
            }
        }
        syn::Expr::Cast(c) => {
            replace_types_in_expr(&mut c.expr, type_map, perm_id, bench_id);
            replace_type_in_type(&mut c.ty, type_map);
        }
        syn::Expr::Index(i) => {
            replace_types_in_expr(&mut i.expr, type_map, perm_id, bench_id);
            replace_types_in_expr(&mut i.index, type_map, perm_id, bench_id);
        }
        syn::Expr::Binary(b) => {
            replace_types_in_expr(&mut b.left, type_map, perm_id, bench_id);
            replace_types_in_expr(&mut b.right, type_map, perm_id, bench_id);
        }
        syn::Expr::Group(g) => {
            replace_types_in_expr(&mut g.expr, type_map, perm_id, bench_id);
        }
        syn::Expr::Field(f) => {
            replace_types_in_expr(&mut f.base, type_map, perm_id, bench_id);
        }
        syn::Expr::ForLoop(fl) => {
            replace_types_in_expr(&mut fl.expr, type_map, perm_id, bench_id);
            replace_types_in_block(&mut fl.body, type_map, perm_id, bench_id);
        }
        syn::Expr::While(w) => {
            replace_types_in_expr(&mut w.cond, type_map, perm_id, bench_id);
            replace_types_in_block(&mut w.body, type_map, perm_id, bench_id);
        }
        syn::Expr::Loop(l) => {
            replace_types_in_block(&mut l.body, type_map, perm_id, bench_id);
        }
        _ => {}
    }
}
