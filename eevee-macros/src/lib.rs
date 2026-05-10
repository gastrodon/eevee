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
}

impl Parse for FnMatrixInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut type_params = Vec::new();
        let mut functions = Vec::new();

        // Parse type parameters (C: T | T, G: T | T, etc.)
        // Try to parse until we get an error
        loop {
            let checkpoint = input.fork();
            match checkpoint.parse::<TypeParam>() {
                Ok(param) => {
                    // Successfully parsed a type param, consume it for real
                    input.parse::<TypeParam>()?;
                    type_params.push(param);
                    // consume optional comma
                    let _ = input.parse::<Token![,]>();
                }
                Err(_) => {
                    // Not a type param, we're done with the type param section
                    break;
                }
            }
        }

        // Parse function definitions with their attributes and visibility
        while !input.is_empty() {
            functions.push(input.parse()?);
        }

        Ok(FnMatrixInput {
            type_params,
            functions,
        })
    }
}

/// Generate multiple function variants by substituting types.
///
/// Example:
/// ```ignore
/// fn_matrix! {
///     C: WConnection,
///     G: NonRecurrent | Recurrent,
///     NN: FeedForward | BinaryFeedForward,
///
///     pub fn foo(g: $G) -> $NN {
///         NN::from_genome(g)
///     }
/// }
/// ```
#[proc_macro]
pub fn fn_matrix(input: TokenStream) -> TokenStream {
    let parsed: FnMatrixInput = match syn::parse(input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };

    let type_map: std::collections::HashMap<_, _> = parsed
        .type_params
        .iter()
        .map(|p| (p.name.to_string(), p))
        .collect();

    let c_variants = type_map.get("C").map(|p| &p.variants);
    let g_variants = type_map.get("G").map(|p| &p.variants);
    let nn_variants = type_map.get("NN").map(|p| &p.variants);

    // At least one type parameter must be present
    if c_variants.is_none() && g_variants.is_none() && nn_variants.is_none() {
        return syn::Error::new_spanned(
            proc_macro2::TokenStream::new(),
            "fn_matrix requires at least one of C, G, or NN type parameters",
        )
        .to_compile_error()
        .into();
    }

    let mut output = quote! {};

    // Generate all combinations using cartesian product
    // Create iterators for each dimension, using references
    let c_iters: Vec<Option<&syn::Type>> = if let Some(variants) = c_variants {
        variants.iter().map(Some).collect()
    } else {
        vec![None]
    };
    let g_iters: Vec<Option<&syn::Type>> = if let Some(variants) = g_variants {
        variants.iter().map(Some).collect()
    } else {
        vec![None]
    };
    let nn_iters: Vec<Option<&syn::Type>> = if let Some(variants) = nn_variants {
        variants.iter().map(Some).collect()
    } else {
        vec![None]
    };

    for c_type in &c_iters {
        for g_type in &g_iters {
            for nn_type in &nn_iters {
                for func in &parsed.functions {
                    let generated = generate_function_variant(func, *c_type, *g_type, *nn_type);
                    output.extend(generated);
                }
            }
        }
    }

    TokenStream::from(output)
}

fn generate_function_variant(
    func: &ItemFn,
    c_type: Option<&syn::Type>,
    g_type: Option<&syn::Type>,
    nn_type: Option<&syn::Type>,
) -> proc_macro2::TokenStream {
    let mut new_fn = func.clone();

    // Generate new function name with type suffixes (only for present types)
    let type_name = |t: &syn::Type| -> String {
        match t {
            syn::Type::Path(p) => p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default(),
            _ => "Unknown".to_string(),
        }
    };

    let orig_name = &func.sig.ident;
    let mut name_parts = vec![orig_name.to_string()];
    if let Some(c) = c_type {
        name_parts.push(type_name(c));
    }
    if let Some(g) = g_type {
        name_parts.push(type_name(g));
    }
    if let Some(nn) = nn_type {
        name_parts.push(type_name(nn));
    }

    let new_name = syn::Ident::new(&name_parts.join("_").to_lowercase(), orig_name.span());

    new_fn.sig.ident = new_name;

    // Replace C, G, NN in the function signature
    replace_types_in_fn(&mut new_fn, c_type, g_type, nn_type);

    quote! {
        #new_fn
    }
}

fn replace_types_in_fn(
    func: &mut ItemFn,
    c_type: Option<&syn::Type>,
    g_type: Option<&syn::Type>,
    nn_type: Option<&syn::Type>,
) {
    // Replace in inputs
    for arg in &mut func.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            replace_type_in_type(&mut pat_type.ty, c_type, g_type, nn_type);
        }
    }

    // Replace in output
    if let ReturnType::Type(_, ty) = &mut func.sig.output {
        replace_type_in_type(ty, c_type, g_type, nn_type);
    }

    // Replace in body
    replace_types_in_block(&mut func.block, c_type, g_type, nn_type);
}

fn replace_type_in_type(
    ty: &mut Box<syn::Type>,
    c_type: Option<&syn::Type>,
    g_type: Option<&syn::Type>,
    nn_type: Option<&syn::Type>,
) {
    if let syn::Type::Path(p) = ty.as_mut() {
        if p.path.segments.len() == 1 {
            let segment = &p.path.segments[0];
            if segment.ident == "C" {
                if let Some(ct) = c_type {
                    **ty = ct.clone();
                }
            } else if segment.ident == "G" {
                if let Some(gt) = g_type {
                    **ty = gt.clone();
                }
            } else if segment.ident == "NN" {
                if let Some(nnt) = nn_type {
                    **ty = nnt.clone();
                }
            }
        }
    }
}

fn replace_types_in_block(
    block: &mut syn::Block,
    c_type: Option<&syn::Type>,
    g_type: Option<&syn::Type>,
    nn_type: Option<&syn::Type>,
) {
    for stmt in &mut block.stmts {
        replace_types_in_stmt(stmt, c_type, g_type, nn_type);
    }
}

fn replace_types_in_stmt(
    stmt: &mut syn::Stmt,
    c_type: Option<&syn::Type>,
    g_type: Option<&syn::Type>,
    nn_type: Option<&syn::Type>,
) {
    match stmt {
        syn::Stmt::Local(local) => {
            if let Some(init) = &mut local.init {
                replace_types_in_expr(&mut init.expr, c_type, g_type, nn_type);
            }
        }
        syn::Stmt::Item(_) => {}
        syn::Stmt::Expr(expr, _) => {
            replace_types_in_expr(expr, c_type, g_type, nn_type);
        }
        syn::Stmt::Macro(_) => {}
    }
}

fn replace_types_in_expr(
    expr: &mut syn::Expr,
    c_type: Option<&syn::Type>,
    g_type: Option<&syn::Type>,
    nn_type: Option<&syn::Type>,
) {
    match expr {
        syn::Expr::Path(p) => {
            // Replace bare C, G, NN references (and in paths like NN::method)
            if let Some(first_segment) = p.path.segments.first() {
                if first_segment.ident == "C" {
                    if p.path.segments.len() == 1 {
                        if let Some(ct) = c_type {
                            *expr = syn::parse2(quote! { #ct }).unwrap_or_else(|_| expr.clone());
                        }
                    } else if let Some(ct) = c_type {
                        // C::method case
                        let remaining = p.path.segments.iter().skip(1).cloned().collect::<Vec<_>>();
                        let mut new_path = match ct {
                            syn::Type::Path(path_type) => path_type.path.clone(),
                            _ => syn::parse2::<syn::Path>(quote! { #ct }).unwrap(),
                        };
                        for seg in remaining {
                            new_path.segments.push(seg);
                        }
                        p.path = new_path;
                    }
                } else if first_segment.ident == "G" {
                    if p.path.segments.len() == 1 {
                        if let Some(gt) = g_type {
                            *expr = syn::parse2(quote! { #gt }).unwrap_or_else(|_| expr.clone());
                        }
                    } else if let Some(gt) = g_type {
                        // G::method case
                        let remaining = p.path.segments.iter().skip(1).cloned().collect::<Vec<_>>();
                        let mut new_path = match gt {
                            syn::Type::Path(path_type) => path_type.path.clone(),
                            _ => syn::parse2::<syn::Path>(quote! { #gt }).unwrap(),
                        };
                        for seg in remaining {
                            new_path.segments.push(seg);
                        }
                        p.path = new_path;
                    }
                } else if first_segment.ident == "NN" {
                    // Replace NN in paths like NN::method
                    if let Some(nnt) = nn_type {
                        let remaining = p.path.segments.iter().skip(1).cloned().collect::<Vec<_>>();
                        if !remaining.is_empty() {
                            // Reconstruct as Type::remaining_path
                            let mut new_path = match nnt {
                                syn::Type::Path(path_type) => path_type.path.clone(),
                                _ => syn::parse2::<syn::Path>(quote! { #nnt }).unwrap(),
                            };
                            for seg in remaining {
                                new_path.segments.push(seg);
                            }
                            p.path = new_path;
                        } else {
                            *expr = syn::parse2(quote! { #nnt }).unwrap_or_else(|_| expr.clone());
                        }
                    }
                }
            }
        }
        syn::Expr::Call(call) => {
            replace_types_in_expr(&mut call.func, c_type, g_type, nn_type);
            for arg in &mut call.args {
                replace_types_in_expr(arg, c_type, g_type, nn_type);
            }
        }
        syn::Expr::MethodCall(mc) => {
            replace_types_in_expr(&mut mc.receiver, c_type, g_type, nn_type);
            for arg in &mut mc.args {
                replace_types_in_expr(arg, c_type, g_type, nn_type);
            }
        }
        syn::Expr::Block(b) => {
            replace_types_in_block(&mut b.block, c_type, g_type, nn_type);
        }
        _ => {}
    }
}
