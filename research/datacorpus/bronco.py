import xml.etree.ElementTree as ET

# Parse the XML data
root = ET.parse(
    "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\datensets\\bronco\\BRONCO150.xml"
).getroot()

# Iterate through the passages
for passage in root.iter("passage"):
    text = passage.find("text").text

    # Iterate through the annotations
    for annotation in passage.findall("annotation"):
        annotation_id = annotation.get("id")
        annotation_type = annotation.find('infon[@key="type"]').text
        annotation_text = annotation.find("text").text
        offset = int(annotation.find("location").get("offset"))
        length = int(annotation.find("location").get("length"))

        print(
            f"Annotation ID: {annotation_id}, Type: {annotation_type}, Text: {annotation_text}, Offset: {offset}, Length: {length}"
        )
