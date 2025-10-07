using UnityEngine;
using Unity.Barracuda;

public class YoloOutputParser
{
    public void ParseOutput(Tensor output, float confThreshold = 0.3f)
    {
        int numBoxes = output.shape.width;
        int numValues = output.shape.channels;

        for (int i = 0; i < numBoxes; i++)
        {
            float conf = output[0, 0, i, 4];
            if (conf < confThreshold) continue;

            int classId = Mathf.RoundToInt(output[0, 0, i, 5]);
            Debug.Log($"Detected class {classId} with confidence {conf:F2}");
        }
    }
}
