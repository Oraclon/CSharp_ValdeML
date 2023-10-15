using System;
using System.ComponentModel.DataAnnotations;
using Microsoft.VisualBasic;

namespace ValdeML
{   
    class Program
    {
        static void Main(string[] args)
        {
            Model model = new Model(Errors.LogLoss);
            model.SetLearningRate(.4);

            Layer Layer1 = new Layer(1, 1, Activation.Tanh, Optimizer.None);
            Layer Layer2 = new Layer(1, 2, Activation.Sigmoid, Optimizer.None);

            DatasetMultFeatures data = new DatasetMultFeatures();
            data.Build(100000, 64, 2, "zscore", true);
            double[][] xx = data.dataset.Select(x => x.input).ToArray();
            double[][] test  = Transposer2.TransposeList(xx);
            double[][] test1 = Transposer2.TransposeList(test); 
            MMODEL[][] batches = data.batches;

            while (model.Error >= 0)
            {
                model.Epoch++;
                for (model.BatchId = 0; model.BatchId < batches.Length; model.BatchId++)
                {
                    MMODEL[] batch = batches[model.BatchId];
                    model.BatchSize = batch.Length;
                    double[][] inputs = batch.Select(x => x.input).ToArray();
                    double[] targets = batch.Select(x => x.target).ToArray();

                    Layer1.TrainNodes(inputs);
                    Layer2.TrainNodes(Layer1.LayerActivations);

                    model.UpdateError(Layer2.LayerDerivatives, targets);

                    Layer2.GetNodesDerivs(model.ErrorDerivs, Layer1.LayerActivations);
                    Layer1.GetNodesDerivs(Layer2.NodeDerivs, inputs);

                    Layer2.UpdateNodes(model);
                    Layer1.UpdateNodes(model);
                }
                string msg = $"{model.Epoch}, {model.Error}";
                Console.WriteLine(msg);
            }
        }
    }
}
