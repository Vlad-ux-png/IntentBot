using System;
using System.Text;
using Microsoft.ML;
using IntentBot;

namespace IntentBot;

class Program
{
    static void Main()
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.InputEncoding = Encoding.UTF8;

        var mlContext = new MLContext(seed: 1);

        IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(path: "data.csv", hasHeader: true, separatorChar: ',');

        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(ModelInput.Label))
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(ModelInput.Text)))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

        var model = pipeline.Fit(dataView);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

        while (true)
        {
            Console.Write("> ");
            string? input = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(input)) continue;

            if (input.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            var sampleData = new ModelInput { Text = input.Trim() };
            var result = predictionEngine.Predict(sampleData);

            string response = result.PredictedLabel switch
            {
                "Greeting" => "How I can really help you?",
                "Time"     => $"The current time is: {DateTime.Now:HH:mm:ss}",
                "About"    => "I am IntentBot.",
                "Goodbye"  => "Goodbye! Have a great day!",
                _          => "How I can really help you?"
            };

            Console.WriteLine($"[Bot] {response}");
        }
    }
}