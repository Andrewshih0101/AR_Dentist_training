using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;
using UnityEditor.Experimental.GraphView;

public class CameraYoloDetector : MonoBehaviour
{
    // 攝影機顯示
    private WebCamTexture camTexture;
    public RawImage background;
    public AspectRatioFitter fit;

    // YOLO 模型
    public NNModel modelAsset;   // 放 YOLOv8 的 .nn 檔
    private Model runtimeModel;
    private IWorker worker;

    // 模型輸入大小 (依照匯出的 onnx 模型設定，通常 640x640)
    private const int INPUT_SIZE = 640;

    void Start()
    {
        // 啟動攝影機
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length > 0)
        {
            camTexture = new WebCamTexture(devices[0].name, Screen.width, Screen.height);
            camTexture.Play();
            background.texture = camTexture;
        }

        // 載入 YOLO 模型
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, runtimeModel);
    }

    void Update()
    {
        if (camTexture == null || !camTexture.didUpdateThisFrame) return;

        // 更新背景畫面
        float ratio = (float)camTexture.width / (float)camTexture.height;
        fit.aspectRatio = ratio;

        // 每隔幾幀做一次推理
        if (Time.frameCount % 10 == 0)
        {
            RunYolo(camTexture);
        }
    }
    Texture2D ResizeTexture(Texture2D source, int width, int height)
    {
        RenderTexture rt = RenderTexture.GetTemporary(width, height);
        Graphics.Blit(source, rt);

        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D result = new Texture2D(width, height, TextureFormat.RGB24, false);
        result.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        result.Apply();

        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(rt);

        return result;
    }
    void RunYolo(WebCamTexture camTex)
    {
        // 1. 將 WebCamTexture 轉成 Texture2D
        Texture2D snap = new Texture2D(camTex.width, camTex.height, TextureFormat.RGB24, false);
        snap.SetPixels(camTex.GetPixels());
        snap.Apply();

        // 2. Resize 並建立 Tensor (直接在 Tensor 內部 resize)
        // INPUT_SIZE = 640 (模型輸入尺寸)
        Texture2D resized = ResizeTexture(snap, INPUT_SIZE, INPUT_SIZE);
        Tensor input = new Tensor(resized, 3);

        // 3. 推理
        worker.Execute(input);
        Tensor output = worker.PeekOutput();

       // Debug.Log("YOLO output shape: " + output.shape);
        YoloOutputParser parser = new YoloOutputParser();
        parser.ParseOutput(output, 0.05f);
        // TODO: 在這裡加輸出解碼與畫框

        // 4. 釋放 Tensor
        input.Dispose();
        output.Dispose();
    }


    private void OnDestroy()
    {
        worker?.Dispose();
    }
}
