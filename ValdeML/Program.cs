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
            demodata.Build(1000000, 256, 2, "zscore", true);

            MMODEL[][] to_train = demodata.batches.Skip(0).Take(demodata.batches.Length - 4).ToArray();
            MMODEL[][] to_eval = demodata.batches.Skip(demodata.batches.Length - 4).ToArray();

            Grad grad = new Grad();
            grad.a = .4;
            BCM lrs = new BCM();
            lrs.Train(grad, to_train, true);

            for (int i = 0; i < to_eval.Length; i++)
            {
                MMODEL[] batch = to_eval[i];
                for (int j = 0; j < batch.Length; j++)
                {
                    double pred = lrs.Predict(grad, batch[j].input);
                    double targ = batch[j].target;
                }
            }

            
        }
    }
}
