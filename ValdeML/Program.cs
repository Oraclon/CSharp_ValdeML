using System;
using System.ComponentModel.DataAnnotations;
using Microsoft.VisualBasic;

namespace ValdeML
{   
    
    public class Layer
    {
        #region Layer Constructor
        public Layer(int layer_size, int layer_id, Activation layer_activation, Optimizer Optimizer) 
        {
            
        }
        #endregion
        #region Layer Variables
        Node[] Nodes                    { get; set; }
        public double[][] Predictions   { get; set; }
        public double[][] Activations   { get; set; }
        public double[][] ActDerivs     { get; set; }
        public double[][] NodeDerivs    { get; set; }
        public double[][] NodeDerivsPow { get; set; }
        public double[][] NodeDerivsW   { get; set; }
        public string LayerId           { get; set; }
        #endregion
    }
    
    class Program
    {
        static void Main(string[] args)
        {
            Model model = new Model(Errors.LogLoss);
            model.SetLearningRate(.4);

            Layer Layer1 = new Layer(4, 1, Activation.Tanh, Optimizer.None);
            Layer Layer2 = new Layer(1, 3, Activation.Sigmoid, Optimizer.None);

            DatasetMultFeatures data = new DatasetMultFeatures();
            data.Build(100000, 256, 2, "zscore", true);
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

                    
                }
                string msg = $"{model.Epoch}, {model.Error}";
                Console.WriteLine(msg);
            }
        }
    }
}
