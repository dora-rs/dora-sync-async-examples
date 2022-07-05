use dora_node_api::{self, config::DataId, DoraNode};

use common::*;
use dora_tracing::{deserialize_context, init_tracing};
use futures::StreamExt;
use opentelemetry::{
    global,
    trace::{TraceContextExt, Tracer},
    Context,
};
use std::sync::Arc;

#[tokio::main(worker_threads = 5)]
async fn main() -> eyre::Result<()> {
    let model = Arc::new(load_model_tract());
    // let mut handles = Vec::with_capacity(40);
    let class_labels = Arc::new(get_imagenet_labels());

    let node = DoraNode::init_from_env().await?;

    let mut inputs = node.inputs().await?;
    node.send_output(&DataId::from("ready".to_owned()), b"")
        .await?;
    let tracer = init_tracing("tokio.spawn").unwrap();
    let input = inputs.next().await.unwrap();
    let string_context = String::from_utf8_lossy(&input.data);
    let context = deserialize_context(&string_context);
    for call_id in 0..100 {
        let _span = tracer.start_with_context(format!("tokio.spawn.{call_id}"), &context);
        let input = match inputs.next().await {
            Some(input) => input,
            None => {
                println!("None");
                continue;
            }
            _ => eyre::bail!(""),
        };
        let call_id = call_id.clone();
        let model = model.clone();
        let class_labels = class_labels.clone();
        //        let (send, recv) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let _context = Context::current_with_span(_span);
            let tracer = global::tracer("name");
            let __span = tracer.start_with_context("tokio-spawn", &_context);
            // run the model on the input
            let image = preprocess(input.data);
            //let input_tensor_values = vec![image];
            let results = run(&model, image);
            // find and display the max value with its index
            let best_result = postprocess(results, &class_labels);
        });
    }

    Ok(())
}
