# 🛣️ Road Damage Detection (YOLOv11)

This project detects and classifies different types of road damages using **YOLOv11**.
It uses a pretrained model (`best.pt`) trained on the [Road Damage dataset](https://universe.roboflow.com/khumbu/road-damage-rwhhe) from Roboflow.

---

## 📂 Repository Contents
- `roadsafe_work.ipynb` — main notebook with step-by-step workflow
- `best.pt` — pretrained YOLOv11 weights (best checkpoint)
- `.env` — store your Roboflow API key here (`ROBOFLOW_API_KEY=xxxxx`)

---

## 🚀 How to Run

1. **Open the notebook**
   - Launch `roadsafe_work.ipynb` in Google Colab (recommended) or locally.

2. **Create a `.env` file** next to the notebook with your Roboflow API key:
   ```bash
   ROBOFLOW_API_KEY=your_key_here

3. **Run the notebook. It will:

Install dependencies (ultralytics, roboflow, opencv, etc.)

Download the Road Damage dataset directly from Roboflow

Load the pretrained model (best.pt)

Run validation & test on the dataset

Generate predictions on validation/test images

All outputs are written locally under ./runs/ (no Google Drive required).
You can download results from Colab’s left file pane (or your local file explorer).



**🏷️ Class Mapping

YOLO outputs numeric class IDs (0–4). They correspond to:

0 → D00 (Longitudinal cracks)
Cracks running parallel to the road centerline.

1 → D10 (Transverse cracks)
Cracks perpendicular to the road centerline.

2 → D20 (Alligator cracks)
Interconnected cracks forming a pattern like alligator skin.

3 → D40 (Potholes)
Bowl-shaped depressions in the pavement.

4 → Other / Ignored
Damage or marks not fitting the above categories.


**📊 Outputs

Metrics and plots are saved automatically in the ./runs/ folder created by YOLO.

Predictions (annotated images) are also saved there, for example:

runs/.../pred_valid/ → validation set predictions

runs/.../pred_test/ → test set predictions

Each image contains bounding boxes labeled with the corresponding class ID (0–4).
Use the mapping above to interpret them.


**📌 Notes

The provided best.pt is the best checkpoint from training and can be reused without retraining.

If you want to retrain on the dataset, uncomment the training cell in the notebook.

Ensure your Roboflow API key is active; otherwise the dataset download will fail.
