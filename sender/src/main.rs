use dora_node_api::{self, config::DataId, DoraNode};
use dora_tracing::init_tracing;
use dora_tracing::serialize_context;
use futures::StreamExt;
use image::EncodableLayout;
use std::path::Path;
use std::time::Duration;

use opentelemetry::{
    trace::{TraceContextExt, Tracer},
    Context,
};
use std::env::var;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let concurrencies = var("CONCURRENCIES")
        .unwrap_or("10".to_string())
        .parse::<u64>()?;
    let iterations = var("ITERATIONS")
        .unwrap_or("10".to_string())
        .parse::<u64>()?;
    let wait_interval = var("WAIT_ITERATIONS_IN_SECS")
        .unwrap_or("5".to_string())
        .parse::<u64>()?;

    let node = DoraNode::init_from_env().await?;

    let mut interval = tokio::time::interval(Duration::from_secs(wait_interval));

    let img = image::open(&Path::new("./data/image.jpg"))?;
    let img = img.as_rgb8().unwrap();

    // let img_width = img.dimensions().0;
    // let img_height = img.dimensions().1;

    // println!("width: {img_width}, height: {img_height}");

    let bytes = img.as_bytes();

    let mut stream = node.inputs().await?;

    // make sure that every node is ready before sending data.
    for _ in 0..node.node_config().inputs.len() {
        let _input = stream.next().await.unwrap();
    }

    let tracer = init_tracing("sender").unwrap();
    let span = tracer.start("root-sender");
    let context = Context::current_with_span(span);

    let context_output = DataId::from("context".to_owned());

    let image_output = DataId::from("image".to_owned());
    for _ in 0..iterations {
        interval.tick().await;
        let span = tracer.start_with_context("child-sender", &context);
        let context = Context::current_with_span(span);
        let string_context = serialize_context(&context);
        node.send_output(&context_output, string_context.as_bytes())
            .await?;
        for _ in 0..concurrencies {
            node.send_output(&image_output, bytes).await?;
        }
    }

    Ok(())
}
