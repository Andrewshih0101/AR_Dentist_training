using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;

public class ScriptsCamera : MonoBehaviour
{
    private WebCamTexture backCam;
    private bool camAvailable;

    public RawImage background;
    public NNModel modelAsset;

    private IWorker worker;
    private RenderTexture tempRT;

    // --- 參數設定 ---
    public int targetDim = 640;
    public float confidenceThreshold = 0.5f;
    public float nmsThreshold = 0.4f;

    public string[] classNames = new string[]
    {
        "head", "syringe"
    };

    void Start()
    {
        // --- 啟動攝影機 ---
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.LogError("No camera detected.");
            return;
        }

        backCam = new WebCamTexture(devices[0].name, 640, 480);
        backCam.Play();
        background.texture = backCam;
        camAvailable = true;

        // --- 初始化模型 ---
        var model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
        tempRT = new RenderTexture(targetDim, targetDim, 24);
    }

    void Update()
    {
        if (!camAvailable) return;

        if (backCam.didUpdateThisFrame && backCam.isPlaying)
        {
            CorrectDisplayAspect();
            RunYolo(backCam);
        }
    }

    // --- 修正比例與旋轉 ---
    void CorrectDisplayAspect()
    {
        if (background == null || backCam.width <= 16) return;

        float imageAspect = (float)backCam.width / backCam.height;
        float containerAspect = 1080f / 1920f;
        RectTransform rect = background.rectTransform;

        rect.localEulerAngles = new Vector3(0, 0, -backCam.videoRotationAngle);

        if (imageAspect > containerAspect)
            rect.localScale = new Vector3(1, containerAspect / imageAspect, 1);
        else
            rect.localScale = new Vector3(imageAspect / containerAspect, 1, 1);
    }

    // --- 執行推論 ---
    void RunYolo(WebCamTexture webCam)
    {
        Graphics.Blit(webCam, tempRT);
        using (Tensor input = new Tensor(tempRT, 3))
        {
            worker.Execute(input);
            Tensor output = worker.PeekOutput();
            ProcessOutput(output);
            output.Dispose();
        }
    }

    // --- 處理輸出 ---
    void ProcessOutput(Tensor output)
    {
        List<Rect> boxes = new List<Rect>();
        List<float> confidences = new List<float>();
        List<int> classIndices = new List<int>();

        int classCount = classNames.Length;
        int predictionsCount = output.shape.width;

        for (int i = 0; i < predictionsCount; i++)
        {
            float maxConf = 0f;
            int bestClass = 0;

            for (int j = 0; j < classCount; j++)
            {
                float conf = output[0, 0, i, 4 + j];
                if (conf > maxConf)
                {
                    maxConf = conf;
                    bestClass = j;
                }
            }

            if (maxConf > confidenceThreshold)
            {
                float cx = output[0, 0, i, 0];
                float cy = output[0, 0, i, 1];
                float w = output[0, 0, i, 2];
                float h = output[0, 0, i, 3];

                float x = cx - w / 2;
                float y = cy - h / 2;

                boxes.Add(new Rect(x, y, w, h));
                confidences.Add(maxConf);
                classIndices.Add(bestClass);
            }
        }

        if (boxes.Count > 0)
        {
            int[] picked = NonMaxSuppression(boxes, confidences, nmsThreshold);
            Debug.Log("--- YOLO 辨識結果 ---");
            foreach (var i in picked)
            {
                Debug.Log($"偵測到: {classNames[classIndices[i]]}, 信心度: {confidences[i]:P1}");
            }
        }
    }

    // --- NMS 與 IOU ---
    int[] NonMaxSuppression(List<Rect> boxes, List<float> scores, float threshold)
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

            indices.RemoveAll(i => CalculateIOU(boxes[current], boxes[i]) > threshold);
        }

        return picked.ToArray();
    }

    float CalculateIOU(Rect a, Rect b)
    {
        float xA = Mathf.Max(a.xMin, b.xMin);
        float yA = Mathf.Max(a.yMin, b.yMin);
        float xB = Mathf.Min(a.xMax, b.xMax);
        float yB = Mathf.Min(a.yMax, b.yMax);

        float inter = Mathf.Max(0, xB - xA) * Mathf.Max(0, yB - yA);
        float areaA = a.width * a.height;
        float areaB = b.width * b.height;
        return inter / (areaA + areaB - inter);
    }

    void OnDestroy()
    {
        worker?.Dispose();
        if (backCam != null) backCam.Stop();
        if (tempRT != null) tempRT.Release();
    }
}
