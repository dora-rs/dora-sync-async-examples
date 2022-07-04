use dora_node_api::{self, config::DataId, DoraNode};
use dora_tracing::init_tracing;
use dora_tracing::serialize_context;
use futures::StreamExt;
use image::EncodableLayout;
use image::ImageBuffer;
use image::Rgb;
use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use opentelemetry::{
    global,
    trace::{TraceContextExt, Tracer},
    Context,
};
static INPUT_PATH: &str = "/home/peter/Documents/CONTRIB/cpu-bound-async/image2.jpg";

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let node = DoraNode::init_from_env().await?;

    let mut interval = tokio::time::interval(Duration::from_millis(20));
    // sleep(Duration::from_secs(0.5));

    let image_output = DataId::from("image".to_owned());
    let img = image::open(&Path::new(INPUT_PATH))?;
    let img = img.as_rgb8().unwrap();
    //let luma = img.as_rgba16().unwrap();

    let img_width = img.dimensions().0;
    let img_height = img.dimensions().1;

    println!("width: {img_width}, height: {img_height}");

    let bytes = img.as_bytes();
    let _test: ImageBuffer<Rgb<u8>, &[u8]> =
        image::ImageBuffer::from_raw(img_width, img_height, bytes).unwrap();
    _test.save("output2.jpg")?;

    let mut stream = node.inputs().await?;
    let input = stream.next().await.unwrap();
    let input = stream.next().await.unwrap();
    let input = stream.next().await.unwrap();
    let tracer = init_tracing("sender").unwrap();
    let span = tracer.start("root-sender");
    let context = Context::current_with_span(span);
    let string_context = serialize_context(&context);
    node.send_output(&image_output, string_context.as_bytes())
        .await?;
    for _ in 0..40 {
        interval.tick().await;
        node.send_output(&image_output, bytes).await?;
    }

    //interval.tick().await;
    //}

    Ok(())
}
