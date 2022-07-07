cd ../dora
cargo build  --manifest-path coordinator/Cargo.toml --release
cd ../dora-sync-async-examples
cp ../dora/target/release/dora-coordinator ./bin/dora-coordinator
cargo build --all --release && ./bin/dora-coordinator run dataflow.yml
