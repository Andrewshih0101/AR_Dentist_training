using UnityEngine;
using UnityEngine.UI; // **新增**: 引用 UI 命名空間
using Unity.Barracuda;
using System.IO;
using System.Collections.Generic;

public class FernQuickTest2 : MonoBehaviour
{
    [Header("UI 設定")] // **新增區塊**
    [Tooltip("將場景中的 Raw Image 物件拖曳到這裡")]
    public RawImage displayImage; // **新增**: 用來顯示攝影機畫面的 RawImage

    [Header("模型設定")]
    [Tooltip("將 ONNX 模型檔案 (NNModel) 從 Project 視窗拖曳到這裡")]
    public NNModel modelAsset;

    // ... (其他變數宣告和之前一樣) ...
    [Tooltip("模型輸入圖片的目標維度")]
    public int targetDim = 640;

    [Header("辨識參數")]
    [Tooltip("信心度閾值，只顯示高於此信心的結果")]
    [Range(0, 1)]
    public float confidenceThreshold = 0.5f;

    [Tooltip("非極大值抑制 (NMS) 的重疊閾值")]
    [Range(0, 1)]
    public float nmsThreshold = 0.4f;


    [Header("類別名稱")]
    public string[] classNames = new string[] {
        "Flower_Hibiscus", "Flower_PeaceLily", "Fruit_Tomato", "Leaf_Areca",
        "Leaf_Croton", "Leaf_Curry", "Leaf_Fiddle", "Leaf_Hibiscus",
        "Leaf_MoneyPlant", "Leaf_PeaceLily", "Leaf_SnakePlant", "Leaf_Tomato",
        "Plant_Areca", "Plant_Croton", "Plant_Curry", "Plant_Fiddle",
        "Plant_Hibiscus", "Plant_MoneyPlant", "Plant_PeaceLily",
        "Plant_SnakePlant", "Plant_Tomato", "Rose_Flower", "Rose_Leaf",
        "Rose_plant"
    };

    private IWorker engine;
    private WebCamTexture webcamTexture;
    private RenderTexture tempRenderTexture;

    void Start()
    {
        var model = ModelLoader.Load(modelAsset);
        engine = WorkerFactory.CreateWorker(model, WorkerFactory.Device.GPU);
        webcamTexture = new WebCamTexture();
        webcamTexture.Play();
        tempRenderTexture = new RenderTexture(targetDim, targetDim, 24);
        
        // **新增**: 將攝影機畫面連接到 RawImage 上
        if (displayImage != null)
        {
            displayImage.texture = webcamTexture;
        }
    }

    void Update()
    {
        if (webcamTexture.didUpdateThisFrame && webcamTexture.isPlaying)
        {
            // **新增**: 修正畫面長寬比與旋轉，避免影像被拉伸或顛倒
            CorrectDisplayAspect();
            
            DetectObjects();
        }
    }

    // **新增函式**: 用來修正畫面顯示
    void CorrectDisplayAspect()
    {
        if (displayImage == null || webcamTexture.width <= 16) return;

        float imageAspect = (float)webcamTexture.width / webcamTexture.height;
        
        // **第 2 處修改**: 將比較的對象從螢幕比例改為固定的 1080x1920 比例
        float containerAspect = 1080f / 1920f;

        RectTransform rect = displayImage.rectTransform;

        // 根據攝影機畫面與容器的比例來調整，避免畫面拉伸
        if (imageAspect > containerAspect)
        {
            float scale = containerAspect / imageAspect;
            rect.localEulerAngles = new Vector3(0, 0, -webcamTexture.videoRotationAngle);
            rect.localScale = new Vector3(1, scale, 1);
        }
        else
        {
            float scale = imageAspect / containerAspect;
            rect.localEulerAngles = new Vector3(0, 0, -webcamTexture.videoRotationAngle);
            rect.localScale = new Vector3(scale, 1, 1);
        }
    }


    void DetectObjects()
    {
        // ... (辨識相關的程式碼完全不變) ...
        Graphics.Blit(webcamTexture, tempRenderTexture);
        Tensor inputTensor = new Tensor(tempRenderTexture, 3);
        engine.Execute(inputTensor);
        Tensor outputTensor = engine.PeekOutput();
        ProcessOutput(outputTensor);
        inputTensor.Dispose();
        outputTensor.Dispose();
    }

    void ProcessOutput(Tensor output)
    {
        // ... (結果處理的程式碼完全不變) ...
        List<Rect> boxes = new List<Rect>();
        List<float> confidences = new List<float>();
        List<int> classIndices = new List<int>();

        int classCount = classNames.Length;
        int predictionsCount = output.shape.width; 

        for (int i = 0; i < predictionsCount; i++)
        {
            float maxConfidence = 0;
            int bestClassIndex = 0;

            for (int j = 0; j < classCount; j++)
            {
                float currentConfidence = output[0, 0, i, 4 + j];
                if (currentConfidence > maxConfidence)
                {
                    maxConfidence = currentConfidence;
                    bestClassIndex = j;
                }
            }

            if (maxConfidence > confidenceThreshold)
            {
                float cx = output[0, 0, i, 0];
                float cy = output[0, 0, i, 1];
                float w = output[0, 0, i, 2];
                float h = output[0, 0, i, 3];

                float x = cx - w / 2;
                float y = cy - h / 2;

                boxes.Add(new Rect(x, y, w, h));
                confidences.Add(maxConfidence);
                classIndices.Add(bestClassIndex);
            }
        }
        
        if (boxes.Count > 0)
        {
            int[] pickedIndices = NonMaxSuppression(boxes, confidences, nmsThreshold);

            Debug.Log("--- 辨識結果 ---");
            foreach (var index in pickedIndices)
            {
                string className = classNames[classIndices[index]];
                float confidence = confidences[index];
                Debug.Log($"找到物件: {className}, 信心度: {confidence:P1}");
            }
        }
    }
    
    // ... (NMS 和 IOU 函式完全不變) ...
    private int[] NonMaxSuppression(List<Rect> boxes, List<float> scores, float threshold)
    {
        List<int> indices = new List<int>();
        for (int i = 0; i < boxes.Count; i++) indices.Add(i);

        indices.Sort((a, b) => scores[b].CompareTo(scores[a]));

        List<int> picked = new List<int>();
        while (indices.Count > 0)
        {
            int current = indices[0];
            picked.Add(current);
            indices.RemoveAt(0);

            List<int> remaining = new List<int>();
            for(int i=0; i<indices.Count; i++)
            {
                float iou = CalculateIOU(boxes[current], boxes[indices[i]]);
                if (iou <= threshold)
                {
                    remaining.Add(indices[i]);
                }
            }
            indices = remaining;
        }
        return picked.ToArray();
    }
    private float CalculateIOU(Rect boxA, Rect boxB)
    {
        float xA = Mathf.Max(boxA.xMin, boxB.xMin);
        float yA = Mathf.Max(boxA.yMin, boxB.yMin);
        float xB = Mathf.Min(boxA.xMax, boxB.xMax);
        float yB = Mathf.Min(boxA.yMax, boxB.yMax);

        float intersectionArea = Mathf.Max(0, xB - xA) * Mathf.Max(0, yB - yA);
        float boxAArea = boxA.width * boxA.height;
        float boxBArea = boxB.width * boxB.height;

        float iou = intersectionArea / (boxAArea + boxBArea - intersectionArea);
        return iou;
    }
    void OnDestroy()
    {
        engine?.Dispose();
        if (webcamTexture != null)
        {
            webcamTexture.Stop();
        }
        if(tempRenderTexture != null)
        {
            tempRenderTexture.Release();
        }
    }
}