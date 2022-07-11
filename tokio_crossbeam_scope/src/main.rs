use dora_node_api::{self, config::DataId, DoraNode};

use common::*;
use dora_tracing::{deserialize_context, init_tracing};
use futures::StreamExt;

use opentelemetry::trace::Tracer;
use opentelemetry::Context as OtelContext;

use crossbeam_utils::thread;
use std::env::var;
use std::sync::{Arc, Mutex, RwLock};
use tokio::runtime::Builder;
use tokio::sync::mpsc;

fn main() -> eyre::Result<()> {
    let number_of_calls = var("NUMBER_OF_CALLS")
        .unwrap_or("100".to_string())
        .parse::<usize>()?;
    // let model = Arc::new(Mutex::new(load_model_gpu()));
    let context = RwLock::new(OtelContext::new());

    let rt = Builder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();
    let (tx, rx) = mpsc::channel(16);

    let model = load_model_tract();
    let tracer = init_tracing("tokio.crossbeam.scope").unwrap();
    std::thread::spawn(move || {
        rt.block_on(async move {
            let node = DoraNode::init_from_env().await.unwrap();
            let mut inputs = node.inputs().await.unwrap();
            node.send_output(&DataId::from("mounted".to_owned()), b"")
                .await
                .unwrap();

            for _ in 0..number_of_calls {
                let input = match inputs.next().await {
                    Some(input) => input,
                    None => {
                        println!("None");
                        continue;
                    }
                };
                tx.send(input).await.unwrap();
            }
        });
    });

    let model = &model;
    let rx = Arc::new(Mutex::new(rx));

    thread::scope(|s| {
        (0..number_of_calls)
            .into_iter()
            .map(|_| {
                s.spawn(|_| {
                    let input = rx.lock().unwrap().blocking_recv().unwrap();
                    match input.id.as_str() {
                        "image" => {
                            let context = context.read().unwrap();
                            let span =
                                tracer.start_with_context(format!("in_async_thread"), &context);
                            let _results = run(model, &input.data, span);
                        }
                        "context" => {
                            let mut context = context.write().unwrap();
                            let string_context = String::from_utf8_lossy(&input.data);
                            *context = deserialize_context(&string_context);
                        }
                        other => eprintln!("Ignoring unexpected input `{other}`"),
                    }
                })
            })
            .for_each(|handle| handle.join().unwrap())
    })
    .unwrap();
    Ok(())
}
