using System;

namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            DatasetMultFeatures dataset = new DatasetMultFeatures();
            dataset.Build(1000000,  64, 2, "zscore", true);

            Grad grad = new Grad();
            grad.a = .2;
            grad.scalers = dataset.scalers;
            BCM bcm = new BCM();
            MMODEL[][] to_train = dataset.batches.Skip(0).Take(dataset.batches.Length - 10).ToArray();
            MMODEL[][] to_eval = dataset.batches.Skip(dataset.batches.Length - 10).ToArray();
            bcm.Train(grad, to_train);

            for(int i = 0; i< to_eval.Length; i++)
            {
                MMODEL[] batch = to_eval[i];
                for(int j =0; j< batch.Length; j++)
                {
                    int prediction = (int)bcm.Predict(grad, batch[j].input);
                    int tar = (int)batch[j].target;
                }
            }

            
            //lrs.Train(grad, to_train);
        }
    }
}