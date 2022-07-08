use common::lazy_download;
use dora_node_api::{self, config::DataId, DoraNode};
use dora_tracing::init_tracing;
use dora_tracing::serialize_context;
use futures::StreamExt;
use image::EncodableLayout;
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
    let image_url = var("IMAGE_URL").unwrap_or("https://i.imgur.com/M23TqZr.jpeg".to_string());

    let node = DoraNode::init_from_env().await?;

    let mut interval = tokio::time::interval(Duration::from_secs(wait_interval));
    let mut interval_context = tokio::time::interval(Duration::from_secs(1));

    let img_path = lazy_download(&image_url)?;
    let img = image::open(&img_path)?;
    let img = img.as_rgb8().unwrap();

    // let img_width = img.dimensions().0;
    // let img_height = img.dimensions().1;

    // println!("width: {img_width}, height: {img_height}");

    let bytes = img.as_bytes();

    let mut stream = node.inputs().await?;

    // make sure that every node is ready before sending data.
    println!("Waiting for node to be mounted..");
    for _ in 0..node.node_config().inputs.len() {
        let _input = stream.next().await.unwrap();
    }
    println!("Node are mounted. Sending data...");

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
        interval_context.tick().await;
        for _ in 0..concurrencies {
            node.send_output(&image_output, bytes).await?;
        }
    }

    Ok(())
}
