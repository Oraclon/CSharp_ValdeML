using System;

namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            DatasetOneFeature dataset = new DatasetOneFeature();
            dataset.Build(1000000,  128, 1.1, "mean");

            Grad grad = new Grad();
            grad.scaler= dataset.scaler;
            grad.a = .4;
            LRS lrs = new LRS();
            lrs.Train(grad, dataset.batches);
            //grad.UpdateW(dataset.dataset[0].input);

            //LRM lrm = new LRM();
            //lrm.Train(grad, dataset.batches);
            //int testint = 1000000;
            //double[] testinp = grad.ScaleInput(new double[] { testint * 100, testint * 10 });
            //double prediction = lrm.Predict(grad, testinp);
        }
    }
}