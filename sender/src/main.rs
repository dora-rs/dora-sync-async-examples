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

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let node = DoraNode::init_from_env().await?;

    let mut interval = tokio::time::interval(Duration::from_millis(10));

    let image_output = DataId::from("image".to_owned());
    let img = image::open(&Path::new("./data/image2.jpg"))?;
    let img = img.as_rgb8().unwrap();

    // let img_width = img.dimensions().0;
    // let img_height = img.dimensions().1;

    // println!("width: {img_width}, height: {img_height}");

    let bytes = img.as_bytes();

    let mut stream = node.inputs().await?;
    // make sure that every node is ready before sending data.
    let _input = stream.next().await.unwrap();
    let _input = stream.next().await.unwrap();
    // let _input = stream.next().await.unwrap();

    let tracer = init_tracing("sender").unwrap();
    let span = tracer.start("root-sender");
    let context = Context::current_with_span(span);
    let string_context = serialize_context(&context);

    node.send_output(&image_output, string_context.as_bytes())
        .await?;
    for _ in 0..100 {
        //interval.tick().await;
        node.send_output(&image_output, bytes).await?;
    }

    Ok(())
}
