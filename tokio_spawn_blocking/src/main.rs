use common::*;
use dora_node_api::{self, config::DataId, DoraNode};
use dora_tracing::{deserialize_context, init_tracing};
use eyre::Context;
use futures::StreamExt;
use opentelemetry::trace::Tracer;
use std::sync::Arc;
use tokio::runtime::Builder;

fn main() -> eyre::Result<()> {
    // let model = Arc::new(Mutex::new(load_model_gpu()));
    let model = Arc::new(load_model_tract());
    // let mut handles = Vec::with_capacity(40);
    let rt = Builder::new_current_thread()
        .enable_all()
        .worker_threads(1)
        .max_blocking_threads(5)
        .build()
        .unwrap();
    rt.block_on(async move {
        let node = DoraNode::init_from_env()
            .await
            .wrap_err("Fail to load dora node")?;

        let mut inputs = node.inputs().await?;
        node.send_output(&DataId::from("ready".to_owned()), b"")
            .await?;
        let tracer = init_tracing("tokio.spawn.blocking")?;
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

            tokio::task::spawn_blocking(move || {
                let _result = run(&model, &input.data, span);
            });
        }
        Ok::<_, eyre::ErrReport>(())
    })
    .unwrap();
    Ok(())
}
