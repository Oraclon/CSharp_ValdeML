using System;

namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            DatasetOneFeature dataset = new DatasetOneFeature();
            dataset.Build(1000000,  256, 2, "zscore");

            Grad grad = new Grad();
            grad.scaler= dataset.scaler;
            grad.a = .2;
            BCS lrs = new BCS();
            SMODEL[][] to_train = dataset.batches.Skip(0).Take(dataset.batches.Length - 10).ToArray();
            SMODEL[][] to_eval = dataset.batches.Skip(dataset.batches.Length - 10).ToArray();
            lrs.Train(grad, to_train);
        }
    }
}