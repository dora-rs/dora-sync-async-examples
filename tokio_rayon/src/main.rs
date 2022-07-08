use dora_node_api::{self, config::DataId, DoraNode};

use common::*;
use dora_tracing::{deserialize_context, init_tracing};
use futures::StreamExt;

use opentelemetry::{
    trace::{TraceContextExt, Tracer},
    Context,
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::{Arc, Mutex};
use tokio::runtime::Builder;
use tokio::sync::mpsc;

fn main() -> eyre::Result<()> {
    // let model = Arc::new(Mutex::new(load_model_gpu()));
    let model = load_model_tract();

    let model = &model;

    rayon::ThreadPoolBuilder::new()
        .num_threads(5)
        .build_global()
        .unwrap();
    let rt = Builder::new_current_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();
    let (tx, rx) = mpsc::channel(16);
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

            for _ in 0..100 {
                let input = match inputs.next().await {
                    Some(input) => input,
                    None => {
                        println!("None");
                        continue;
                    }
                };
                tx.send(input.data).await.unwrap();
            }
        });
    });
    let context = rx_context.blocking_recv().unwrap();
    let rx = Arc::new(Mutex::new(rx));
    (0..100)
        .into_par_iter()
        .map(|_: i32| {
            let data = rx.lock().unwrap().blocking_recv().unwrap();
            let _span = tracer.start_with_context(format!("in_sync_thread"), &context);
            let _context = Context::current_with_span(_span);
            let image = preprocess(&data);
            let _results = run(model, image);
        })
        .collect::<Vec<_>>();
    Ok(())
}
