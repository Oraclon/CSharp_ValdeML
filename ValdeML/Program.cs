using System;

namespace ValdeML
{
    enum Fruit
    {
        orange,
        grapefruit
    }
    class Program
    {
        static void Main(string[] args)
        {
            DatasetMultFeatures demodata = new DatasetMultFeatures();
            demodata.Build(100000, 128, 2, "zscore", false);

            MMODEL[][] to_train = demodata.batches.Skip(0).Take(demodata.batches.Length - 4).ToArray();
            MMODEL[][] to_eval = demodata.batches.Skip(demodata.batches.Length - 4).ToArray();

            Grad grad = new Grad();
            grad.a = .4;
            LRM lrs = new LRM();
            lrs.Train(grad, to_train);

            double pred = lrs.Predict(grad, to_eval[0][0].input);
            double targ = to_eval[0][0].target;
        }
    }
}
