{
  description = "eevee — Generalized NeuroEvolution toolkit based on NEAT";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, crane, fenix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;

        toolchain = fenix.packages.${system}.latest.toolchain;

        craneLib = (crane.mkLib pkgs).overrideToolchain toolchain;

        # Include examples/*/data alongside Rust sources so embedded assets
        # (tetris.nes, sentiment word lists, etc.) are present at build time.
        src = lib.cleanSourceWith {
          src = ./.;
          filter = path: type:
            (craneLib.filterCargoSources path type)
            || (builtins.match ".*/examples/[^/]+/data(/.*)?$" path != null);
        };

        commonArgs = {
          inherit src;
          strictDeps = true;
        };

        # Build all dependency crates once with every optional feature enabled,
        # so individual example builds can reuse the same artifact cache.
        cargoArtifacts = craneLib.buildDepsOnly (commonArgs // {
          cargoExtraArgs = "--features approx,parallel";
        });

        buildExample = { name, features ? [ ] }:
          let
            featuresArg =
              if features != [ ]
              then " --features ${lib.concatStringsSep "," features}"
              else "";
          in
          craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts;
            pname = "eevee-example-${name}";
            cargoExtraArgs = "--example ${name}${featuresArg}";
          });

      in
      {
        packages = {
          default = craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts;
          });

          example = {
            xor = buildExample {
              name = "xor";
              features = [ "approx" ];
            };
            sentiment = buildExample {
              name = "sentiment";
            };
          };
        };

        devShells.default = craneLib.devShell {
          packages = [ pkgs.cargo-flamegraph pkgs.gnuplot ];
        };
      }
    );
}
