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
            //DatasetMultFeatures demodata = new DatasetMultFeatures();
            //demodata.Build(1000000, 256, 5, "zscore", true);
            Random r = new Random();

            string path = "/Users/angularnodedeveloper/Projects/datasets/datasets/sales_dataset.csv";
            StreamReader read = new StreamReader(path);
            string[] lines = read.ReadToEnd().Split("\n");
            MMODEL[] dataset = new MMODEL[lines.Length];
            for (int i = 0; i < lines.Length; i++)
            {
                MMODEL model = new MMODEL();
                double[] line = lines[i].Split(",").Select(x => Convert.ToDouble(x)).ToArray();
                model.input = line.Skip(1).ToArray();
                model.target = line[0];
                dataset[i] = model;
            }
            int test1 = dataset.Where(x => x.target == 1).ToArray().Length;
            int test2 = dataset.Where(x => x.target == 0).ToArray().Length;

            var d1= dataset.Where(x => x.target == 1);
            var d2 = dataset.Where(x => x.target == 0).Skip(0).Take(test1);

            IEnumerable<MMODEL> new_dataset = d1.Concat(d2);

            dataset = new_dataset.ToArray();
            ZSCORE scaler = new ZSCORE();
            dataset = scaler.Get(new_dataset.ToArray());
            dataset = dataset.OrderBy(_ => r.Next()).ToArray();
            MMODEL[][] batches = new Batches().Get(dataset, 32);
            MMODEL[][] to_train = batches.Skip(0).Take(batches.Length - 2).ToArray();
            MMODEL[][] to_eval = batches.Skip(batches.Length - 2).ToArray();

            Grad grad = new Grad();
            grad.SetTolerance(10);
            grad.a = .4;
            BCM lrs = new BCM();
            lrs.Train(grad, to_train, true);

            int correct = 0;
            int wrong = 0;
            for (int i = 0; i < to_eval.Length; i++)
            {
                MMODEL[] batch = to_eval[i];
                for (int j = 0; j < batch.Length; j++)
                {
                    double pred = lrs.Predict(grad, batch[j].input);
                    double targ = batch[j].target;

                    if (pred.Equals(targ))
                        correct++;
                    else
                        wrong++;
                }
            }
                string res = $"{correct} {wrong} {(correct * 100) / (correct + wrong)}%";
        }
    }
}
