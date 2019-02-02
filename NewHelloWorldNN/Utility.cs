using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NewHelloWorldNN
{
    public class Utility
    {
        public static string ArrayToString(double[] array, string format)
        {
            string s = "( ";
            for (int i = 0; i < array.Length; i++)
            {
                s += array[i].ToString(format);
                if (i != array.Length - 1)
                {
                    s += ", ";
                }
                else
                {
                    s += " )";
                }
            }
            return s;
        }

        public static string ArrayToString(double[] array)
        {
            string s = "( ";
            for (int i = 0; i < array.Length; i++)
            {
                s += array[i].ToString();
                if (i != array.Length - 1)
                {
                    s += ", ";
                }
                else
                {
                    s += " )";
                }
            }
            return s;
        }
    }
}
