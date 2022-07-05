use dora_node_api::{self, config::DataId, DoraNode};

use common::*;
use dora_tracing::{deserialize_context, init_tracing};
use futures::{StreamExt, TryFutureExt};

use opentelemetry::{
    global,
    trace::{TraceContextExt, Tracer},
    Context,
};
use pollster::FutureExt as _;

use tokio::runtime::Builder;
use tokio::sync::mpsc;

fn main() -> eyre::Result<()> {
    let model = load_model_tract();
    let class_labels = get_imagenet_labels();

    let model = &model;
    let class_labels = &class_labels;

    // Build the runtime for the new thread.
    //
    // The runtime is created before spawning the thread
    // to more cleanly forward errors if the `unwrap()`
    // panics.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(5)
        .build()
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

    pool.scope(|s| {
        for call_id in 0..100 {
            let data = rx.blocking_recv().unwrap();
            let _span = tracer.start_with_context(format!("tokio.rayon.{call_id}"), &context);
            s.spawn(|_| {
                let _context = Context::current_with_span(_span);
                let tracer = global::tracer("name");
                let __span = tracer.start_with_context("tokio-spawn", &_context);
                // run the model on the input
                let image = preprocess(data);
                //let input_tensor_values = vec![image];
                let results = run(model, image);
                // find and display the max value with its index
                let best_result = postprocess(results, class_labels);
            });
        }
    });

    Ok(())
}
