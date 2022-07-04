cd ../dora
cargo build  --manifest-path coordinator/Cargo.toml --release
cd ../cpu-bound-async
cp ../dora/target/release/dora-coordinator bin/dora-coordinator
cargo build --all && ./bin/dora-coordinator run dataflow.yml