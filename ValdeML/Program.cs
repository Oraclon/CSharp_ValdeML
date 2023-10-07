using System;

namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            DatasetOneFeature dataset = new DatasetOneFeature();
            dataset.Build(1000000,  128, 2, "zscore");

            Grad grad = new Grad();
            grad.scaler= dataset.scaler;
            grad.a = .4;
            BCS lrs = new BCS();
            SMODEL[][] to_train = dataset.batches.Skip(0).Take(dataset.batches.Length - 4).ToArray();
            SMODEL[][] to_eval = dataset.batches.Skip(dataset.batches.Length - 4).ToArray();
            lrs.Train(grad, to_train);

            int correct = 0;
            int wrong = 0;
            for(int i = 0; i< to_eval.Length; i++)
            {
                SMODEL[] batch = to_eval[i];
                for(int j= 0; j< batch.Length; j++)
                {
                    int predict = (int)lrs.Prediction(grad, batch[j].input);
                    int target = (int)batch[j].target;
                    if (predict.Equals(target))
                        correct++;
                    else
                        wrong++;
                }
            }
            string res = $"{correct}, {wrong}, {(correct*100)/(correct+wrong)}";
            //int testint = 123400;
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