use common::*;
use dora_node_api::{self, config::DataId, DoraNode};
use dora_tracing::{deserialize_context, init_tracing};
use eyre::Context;
use futures::StreamExt;
use glommio::{executor, Latency, LocalExecutorBuilder, Placement, Shares};
use opentelemetry::trace::Tracer;
use opentelemetry::Context as OtelContext;
use std::env::var;
use std::sync::{Arc, RwLock};

fn main() -> eyre::Result<()> {
    let blocking_threads = var("TOKIO_BLOCKING_THREADS")
        .unwrap_or("50".to_string())
        .parse::<usize>()?;
    let worker_threads = var("TOKIO_WORKER_THREADS")
        .unwrap_or("1".to_string())
        .parse::<usize>()?;
    let number_of_calls = var("NUMBER_OF_CALLS")
        .unwrap_or("100".to_string())
        .parse::<usize>()?;
    // let model = Arc::new(Mutex::new(load_model_gpu()));
    let model = Arc::new(load_model_tract());
    let context = RwLock::new(OtelContext::new());

    let local_ex = LocalExecutorBuilder::default();

    local_ex
        .block_on(async move {
            let node = DoraNode::init_from_env()
                .await
                .wrap_err("Fail to load dora node")?;

            let mut inputs = node.inputs().await?;
            node.send_output(&DataId::from("mounted".to_owned()), b"")
                .await?;
            let tracer = init_tracing("glommio.spawn.blocking")?;
            for _ in 0..number_of_calls {
                let input = match inputs.next().await {
                    Some(input) => input,
                    None => {
                        println!("None");
                        continue;
                    }
                };

                match input.id.as_str() {
                    "image" => {
                        let context = context.read().unwrap();
                        let span = tracer.start_with_context(format!("in_async_thread"), &context);
                        let model = model.clone();

                        glommio::task::spawn_blocking(move || {
                            let _result = run(&model, &input.data, span);
                        });
                    }
                    "context" => {
                        let mut context = context.write().unwrap();
                        let string_context = String::from_utf8_lossy(&input.data);
                        *context = deserialize_context(&string_context);
                    }
                    other => eprintln!("Ignoring unexpected input `{other}`"),
                }
            }
            Ok::<_, eyre::ErrReport>(())
        })
        .unwrap();
    Ok(())
}
