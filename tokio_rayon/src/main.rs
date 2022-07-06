use dora_node_api::{self, config::DataId, DoraNode};

use common::*;
use dora_tracing::{deserialize_context, init_tracing};
use futures::StreamExt;

use opentelemetry::{
    global,
    trace::{TraceContextExt, Tracer},
    Context,
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::{Arc, Mutex};
use tokio::runtime::Builder;
use tokio::sync::mpsc;

fn main() -> eyre::Result<()> {
    let model = Arc::new(Mutex::new(load_model_gpu()));
    let class_labels = get_imagenet_labels();

    let model = &model;
    let class_labels = &class_labels;

    // Build the runtime for the new thread.
    //
    // The runtime is created before spawning the thread
    // to more cleanly forward errors if the `unwrap()`
    // panics.
    rayon::ThreadPoolBuilder::new()
        .num_threads(5)
        .build_global()
        .unwrap();
    let rt = Builder::new_current_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();
    let (tx, mut rx) = mpsc::channel(16);
    let (tx_context, mut rx_context) = mpsc::channel(16);
    let tracer = init_tracing("tokio.rayon").unwrap();
    std::thread::spawn(move || {
        rt.block_on(async move {
            let node = DoraNode::init_from_env().await.unwrap();
            let mut inputs = node.inputs().await.unwrap();
            node.send_output(&DataId::from("ready".to_owned()), b"")
                .await
                .unwrap();
            let input = inputs.next().await.unwrap();
            let string_context = String::from_utf8_lossy(&input.data);
            let context = deserialize_context(&string_context);
            tx_context.send(context).await.unwrap();

            for call_id in 0..100 {
                let input = match inputs.next().await {
                    Some(input) => input,
                    None => {
                        println!("None");
                        continue;
                    }
                };
                tx.send(input.data).await.unwrap();
                // Once all senders have gone out of scope,
                // the `.recv()` call returns None and it will
                // exit from the while loop and shut down the
                // thread.
            }
        });
    });
    let context = rx_context.blocking_recv().unwrap();
    let rx = Arc::new(Mutex::new(rx));
    (0..100)
        .into_par_iter()
        .map(|call_id: i32| {
            let data = rx.lock().unwrap().blocking_recv().unwrap();
            let data = &data;
            rayon::scope(|t| {
                let _span = tracer.start_with_context(format!("tokio.rayon.{call_id}"), &context);
                t.spawn(|_| {
                    let _context = Context::current_with_span(_span);
                    let tracer = global::tracer("name");
                    let __span = tracer.start_with_context("tokio-spawn", &_context);
                    // run the model on the input
                    let image = preprocess_gpu(&data);
                    let results = run_gpu(model.clone(), image);
                    // find and display the max value with its index
                    // let best_result = postprocess(results, class_labels);
                });
            })
        })
        .collect::<Vec<_>>();
    Ok(())
}
