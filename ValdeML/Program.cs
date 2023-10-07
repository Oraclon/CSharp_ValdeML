using System;

namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            DatasetOneFeature dataset = new DatasetOneFeature();
            dataset.Build(1000000,  128, 2, "mean");

            Grad grad = new Grad();
            grad.scaler= dataset.scaler;
            grad.a = .4;
            BCS lrs = new BCS();
            lrs.Train(grad, dataset.batches);
            int testint = 123400;
            //double test_input = grad.SScaleInput(testint);
            //double prediction = lrs.Predict(grad, test_input);


            //LRM lrm = new LRM();
            //grad.UpdateW(dataset.dataset[0].input);
            //lrm.Train(grad, dataset.batches);

            //double[] testinp = grad.ScaleInput(new double[] { testint * 100, testint * 10 });
            //double prediction = lrm.Predict(grad, testinp);
        }
    }
}