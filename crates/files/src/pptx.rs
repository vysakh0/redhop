//! PPTX text extraction: a PPTX is a zip of OOXML; slide text lives in
//! `ppt/slides/slideN.xml` inside `<a:t>` runs. One section per slide
//! (page = slide number).

use std::io::Read;

use quick_xml::events::Event;
use quick_xml::Reader as XmlReader;
use zip::ZipArchive;

use crate::{ExtractError, ExtractedDoc, Section};

/// Cap total uncompressed slide XML we'll read, to bound decompression ("zip
/// bomb") amplification from a small malicious .pptx.
const MAX_SLIDE_XML_BYTES: u64 = 256 * 1024 * 1024;

pub(crate) fn extract_bytes(data: &[u8], source: String) -> Result<ExtractedDoc, ExtractError> {
    let mut zip = ZipArchive::new(std::io::Cursor::new(data))
        .map_err(|e| ExtractError::Parse(format!("pptx zip: {e}")))?;

    let mut slides: Vec<String> = Vec::new();
    for i in 0..zip.len() {
        let name = zip
            .by_index(i)
            .map_err(|e| ExtractError::Parse(format!("pptx zip entry: {e}")))?
            .name()
            .to_string();
        if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
            slides.push(name);
        }
    }
    slides.sort_by_key(|n| slide_num(n));

    let mut sections = Vec::new();
    let mut budget = MAX_SLIDE_XML_BYTES;
    for (idx, name) in slides.iter().enumerate() {
        let mut entry = zip
            .by_name(name)
            .map_err(|e| ExtractError::Parse(format!("pptx slide: {e}")))?;
        // Reject before allocating: a single entry larger than what's left of the
        // uncompressed budget is treated as a bomb.
        if entry.size() > budget {
            return Err(ExtractError::Parse(
                "pptx: uncompressed size exceeds limit (possible zip bomb)".into(),
            ));
        }
        budget -= entry.size();
        let mut xml = String::new();
        entry.read_to_string(&mut xml).map_err(ExtractError::Io)?;
        let text = slide_text(&xml);
        if !text.trim().is_empty() {
            sections.push(Section {
                text,
                page: Some(idx + 1),
                heading: None,
                line: None,
            });
        }
    }
    Ok(ExtractedDoc { source, sections })
}

fn slide_num(name: &str) -> usize {
    name.trim_start_matches("ppt/slides/slide")
        .trim_end_matches(".xml")
        .parse()
        .unwrap_or(0)
}

/// Concatenate the text in every `<a:t>` element of a slide's XML.
fn slide_text(xml: &str) -> String {
    let mut reader = XmlReader::from_str(xml);
    let mut runs: Vec<String> = Vec::new();
    let mut in_t = false;
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) if e.local_name().as_ref() == b"t" => in_t = true,
            Ok(Event::End(e)) if e.local_name().as_ref() == b"t" => in_t = false,
            Ok(Event::Text(t)) if in_t => {
                runs.push(t.unescape().unwrap_or_default().into_owned());
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    runs.join(" ")
}
