using System;

namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            Transposer transposer = new Transposer();
            int s = 20;
            MMODEL[] dataset = new MMODEL[s];
            for(int i = 0; i< s; i++)
            {
                int x = i + 1;
                MMODEL model = new MMODEL();
                model.input= new double[] { x, x*2, x*3, x*4, x*5 };
                model.target = x * 2;
                dataset[i] = model;
            }

            Grad grad = new Grad();
            ZScore scaled = new ZScore();
            grad.scalers = scaled.scalers;
            dataset = scaled.Get(dataset);
        }
    }
}