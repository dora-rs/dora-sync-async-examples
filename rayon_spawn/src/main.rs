use dora_node_api::{self, config::DataId, DoraNode};

use common::*;
use dora_tracing::{deserialize_context, init_tracing};
use futures::StreamExt;
use opentelemetry::trace::Tracer;
use std::sync::Arc;

#[tokio::main(worker_threads = 1)]
async fn main() -> eyre::Result<()> {
    let model = Arc::new(load_model_tract());
    let mut handles = Vec::with_capacity(40);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build()
        .unwrap();
    let node = DoraNode::init_from_env().await?;

    let mut inputs = node.inputs().await?;
    node.send_output(&DataId::from("ready".to_owned()), b"")
        .await?;
    let tracer = init_tracing("rayon.spawn").unwrap();
    let input = inputs.next().await.unwrap();
    let string_context = String::from_utf8_lossy(&input.data);
    let context = deserialize_context(&string_context);
    for _ in 0..100 {
        let input = match inputs.next().await {
            Some(input) => input,
            None => {
                println!("None");
                continue;
            }
        };
        let span = tracer.start_with_context(format!("in_async_thread"), &context);
        let model = model.clone();

        let (send, recv) = tokio::sync::oneshot::channel();
        pool.spawn_fifo(move || {
            let result = run(&model, &input.data, span);
            send.send(result).unwrap();
        });
        handles.push(recv);
    }
    futures::future::join_all(handles).await;
    Ok(())
}
