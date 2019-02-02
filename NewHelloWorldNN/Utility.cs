using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NewHelloWorldNN
{
    public class Utility
    {
        /// <summary>
        /// Formats an array of doubles into a string after rounding to specified format
        /// </summary>
        /// <param name="array">Array to format</param>
        /// <param name="format">format used when evaluating String.Format(string, format)</param>
        /// <returns></returns>
        public static string ArrayToString(double[] array, string format)
        {
            string s = "(";
            for (int i = 0; i < array.Length; i++)
            {
                s += array[i].ToString(format);
                if (i != array.Length - 1)
                {
                    s += ", ";
                }
                else
                {
                    s += ")";
                }
            }
            return s;
        }

        /// <summary>
        /// Formats an array of doubles into a string
        /// </summary>
        /// <param name="array">Array to format</param>
        /// <returns></returns>
        public static string ArrayToString(double[] array)
        {
            string s = "(";
            for (int i = 0; i < array.Length; i++)
            {
                s += array[i].ToString();
                if (i != array.Length - 1)
                {
                    s += ", ";
                }
                else
                {
                    s += ")";
                }
            }
            return s;
        }
    }
}
