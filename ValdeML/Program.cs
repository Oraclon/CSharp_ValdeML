using System;
using System.ComponentModel.DataAnnotations;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.VisualBasic;

namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            Layer layer1 = new Layer(24, Activation.Tanh);
            Layer layer2 = new Layer(24, Activation.Tanh);
            Layer layer3 = new Layer(1, Activation.Sigmoid);
            DatasetMultFeatures data = new DatasetMultFeatures();
            data.Build(100000, 512, 4, "zscore", true);
            Model model = new Model(Errors.LogLoss);
            model.Learning = .4;

            while (model.Error >= 0)
            {
                for (model.BatchId = 0; model.BatchId < data.batches.Length; model.BatchId++)
                {
                    MMODEL[] batch = data.batches[model.BatchId];
                    double[][] inputs = batch.Select(x => x.input).ToArray();
                    double[] targets = batch.Select(x => x.target).ToArray();
                    model.BatchSize = batch.Length;

                    layer1.NodesPredict(inputs);
                    layer2.NodesPredict(layer1.nodeActivations);
                    layer3.NodesPredict(layer2.nodeActivations);

                    model.CalculateError(layer3.nodeActivations, targets);

                    layer3.NodesCalcDeltas(model.ErrorDerivs, layer2.nodeActivations);
                    layer2.NodesCalcDeltas(layer3.nodeDeltas, layer1.nodeActivations);
                    layer1.NodesCalcDeltas(layer2.nodeDeltas, inputs);

                    layer3.NodesUpdate(model);
                    layer2.NodesUpdate(model);
                    layer1.NodesUpdate(model);

                    if (model.Error <= Math.Pow(10, -3))
                        break;
                }
                if (model.Error <= Math.Pow(10, -3))
                {
                    string res = $"{model.Epoch}, {model.BatchId}, {model.Error}";
                    break;
                }
            }
        }
    }
}
