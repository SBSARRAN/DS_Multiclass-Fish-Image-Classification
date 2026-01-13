import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


st.set_page_config(page_title="Fish Classification", layout="centered")

device = torch.device("cpu")
num_classes = 11

Fish_label = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 
    'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel', 
    'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 
    'fish sea_food sea_bass', 'fish sea_food shrimp', 'fish sea_food striped_red_mullet',
      'fish sea_food trout'
]


@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    model.load_state_dict(
        torch.load("Best_Fish_Model.pth", map_location=device)
    )
    model.eval()
    return model

model = load_model()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


st.title("🐟 Fish Species Classification")
st.write(
    "Upload a fish image and get the predicted species "
    "using **EfficientNetB0**."
)

uploaded_file = st.file_uploader(
    "Upload Fish Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    st.markdown("### 🔍 Prediction Result")
    st.success(f"**Fish Species:** {Fish_label[predicted_class.item()]}")
    st.info(f"Trained Model Name :EfficientNetB0")
    st.info(f"**Confidence:** {confidence.item()*100:.2f}%")
