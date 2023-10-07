using System;

namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            DatasetMultFeatures dataset = new DatasetMultFeatures();
            dataset.Build(100000, 128, 1000, "mean");

            Grad grad = new Grad();
            grad.scalers = dataset.scalers;
            grad.UpdateW(dataset.dataset[0].input);
            grad.a = .4;

            LRM lrm = new LRM();
            lrm.Train(grad, dataset.batches);
            int testint = 1000000;
            double[] testinp = grad.ScaleInput(new double[] { testint * 100, testint * 10 });
            double prediction = lrm.Predict(grad, testinp);
        }
    }
}