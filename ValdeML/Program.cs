using System;
using System.ComponentModel.DataAnnotations;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.VisualBasic;

namespace ValdeML
{
    static class DemoDataset
    {

    }

    class Program
    {
        static void Main(string[] args)
        {
            Node node = new Node(1, Activation.Sigmoid);
            DatasetMultFeatures data = new DatasetMultFeatures();
            data.Build(100000, 128, 2, "zscore", true);
            Model model = new Model(Errors.LogLoss);
            while (model.Error >= 0)
            {
                for (model.BatchId = 0; model.BatchId < data.batches.Length; model.BatchId++)
                {
                    MMODEL[] batch = data.batches[model.BatchId];
                    double[][] inputs = batch.Select(x => x.input).ToArray();
                    double[] targets = batch.Select(x => x.target).ToArray();

                    node.NodePredict(inputs);
                }
            }
        }
    }
}
