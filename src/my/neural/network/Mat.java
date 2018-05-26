package my.neural.network;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * This is a helper class for the {@link NeuralNetwork} class that
 * simply holds a matrix and allows us to perform mass operations on
 * all or certain values of the matrix.
 * 
 * @author Hashim Kayani
 */
public class Mat implements Serializable, Cloneable
{
	/**
	 * The amount of rows in the matrix.<br>
	 * {
	 *     {...},
	 *     {...},
	 *     {...},
	 *     {...}
	 * }
	 * <br>
	 * This matrix has 4 rows
	 */
	public final int rows;
	/**
	 * The amount of columns in the matrix.<br>
	 * {
	 *     { {ab}, {cd}, {ef} },
	 *     { {..}, {..}, {..} }
	 * }
	 * <br>
	 * This matrix has 3 columns
	 */
	public final int cols;
	/**
	 * This is the raw double array with the respective rows and
	 * columns. This would be the same as a
	 * <code>new double[rows][cols]</code>.
	 */
	public final double[][] data;
	
	/**
	 * This creates a matrix with the rows and columns specified in
	 * the parameters. It'll be a new 2-dimensional double array
	 * initialized as <code>new double[rows][cols]</code>.
	 * 
	 * @param rows The amount of rows in the matrix
	 * @param cols The amount of columns in the matrix
	 */
	public Mat(int rows, int cols)
	{
		this.rows = rows;
		this.cols = cols;
		this.data = new double[rows][cols];
	}
	
	/**
	 * This creates a new matrix with the specified double array.
	 * Keep in mind that this constructor assumes that the array is
	 * a proper 2-dimensional array. Meaning it'll that the length of
	 * the first row is the same as all other rows.
	 * 
	 * @param data This is the input array for the matrix
	 */
	public Mat(double[][] data)
	{
		this.rows = data.length;
		this.cols = data[0].length;
		this.data = new double[rows][cols];
		// Simple copy of a 2-dimensional double array
		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < cols; c++)
			{
				this.data[r][c] = data[r][c];
			}
		}
	}
	
	/**
	 * This method will randomize all the data for this matrix. What
	 * the values are set to are between <code>-1</code> to
	 * <code>1</code> using the <code>ThreadLocalRandom.current()</code>
	 * method.<br><br><b>NOTE: This modifies the current matrix!</b>
	 * 
	 * @return this
	 */
	public Mat randomize()
	{
		return randomize(ThreadLocalRandom.current());
	}

	
	/**
	 * This method will randomize all the data for this matrix. What
	 * the values are set to are between <code>-1</code> to
	 * <code>1</code> given a random number generator.
	 * <br><br><b>NOTE: This modifies the current matrix!</b>
	 * 
	 * @param rand This is the random number generator to use
	 * 
	 * @return this
	 */
	public Mat randomize(final Random rand)
	{
		return map(new MatFunc()
		{
			@Override
			public double perform(double val, int r, int c)
			{
				return rand.nextDouble() * 2 - 1;
			}
		});
	}
	
	/**
	 * Add each element in a row and column to the corresponding
	 * index in the given matrix.
	 * 
	 * @param mat The other matrix to add to this
	 * 
	 * @return A new Mat with the addition of the elements
	 */
	public Mat add(final Mat mat)
	{
		return new Mat(data).map(new MatFunc()
		{
			@Override
			public double perform(double val, int r, int c)
			{
				return val + mat.data[r][c];
			}
		});
	}
	
	/**
	 * Add the given value to each element in a row and column
	 * 
	 * @param scl The value to add to each element
	 * 
	 * @return A new Mat with the addition of the parameter
	 */
	public Mat add(final double v)
	{
		return new Mat(data).map(new MatFunc()
		{
			@Override
			public double perform(double val, int r, int c)
			{
				return val + v;
			}
		});
	}
	
	/**
	 * Subtract each element in the given matrix from the
	 * corresponding index in this matrix.
	 * 
	 * @param mat The other matrix to subtract from this
	 * 
	 * @return A new Mat with the subtraction of the parameter
	 */
	public Mat subtract(final Mat mat)
	{
		return new Mat(data).map(new MatFunc()
		{
			@Override
			public double perform(double val, int r, int c)
			{
				return val - mat.data[r][c];
			}
		});
	}
	
	/**
	 * Subtract the given value from each element in this matrix
	 * 
	 * @param scl The value to subtract from each element
	 * 
	 * @return A new Mat with the subtraction of the parameter
	 */
	public Mat subtract(final double v)
	{
		return new Mat(data).map(new MatFunc()
		{
			@Override
			public double perform(double val, int r, int c)
			{
				return val - v;
			}
		});
	}
	
	/**
	 * Diagonally flip a copy of this matrix<br>
	 * Code:<br>
	 * <code>
	 * Mat mat = new Mat({ {1, 2}, {3, 4}, {5, 6} });<br>
	 * mat = mat.transpose(); // Returns new Mat<br>
	 * </code>
	 * Mat will transpose like so:<br>
	 * <code>
	 * [<b>1</b>, 2]<br>
	 * [3, <b>4</b>]<br>
	 * [5, 6]<br>
	 * <br>
	 * [<b>1</b>, 3, 5]<br>
	 * [2, <b>4</b>, 6]<br>
	 * </code>
	 * 
	 * @return A new Mat with the transposition of this
	 */
	public Mat transpose()
	{
		return new Mat(cols, rows).map(new MatFunc()
		{
			@Override
			public double perform(double val, int r, int c)
			{
				return data[c][r];
			}
		});
	}
	
	/**
	 * Multiply the given value with each element in this matrix
	 * 
	 * @param scl The value to multiply with each element
	 * 
	 * @return A new Mat with the multiplication of the parameter
	 */
	public Mat mult(final double scl)
	{
		return new Mat(data).map(new MatFunc()
		{
			@Override
			public double perform(double val, int r, int c)
			{
				return val * scl;
			}
		});
	}

	/**
	 * Multiply each element in this matrix with the
	 * corresponding index in the given matrix. This only multiplies
	 * matching elements.
	 * 
	 * @param mat The matrix to multiply from this
	 * 
	 * @return A new Mat with the multiplication of the parameter
	 */
	public Mat elementMult(final Mat mat)
	{
		return new Mat(data).map(new MatFunc()
		{
			@Override
			public double perform(double val, int r, int c)
			{
				return val * mat.data[r][c];
			}
		});
	}

	/**
	 * {@link https://en.wikipedia.org/wiki/Matrix_multiplication}
	 */
	public Mat mult(final Mat mat)
	{
		if(cols != mat.rows) throw new RuntimeException("Rows don't match columns");
		
		return new Mat(rows, mat.cols).map(new MatFunc()
		{
			@Override
			public double perform(double val, int r, int c)
			{
				double sum = 0;
				for(int i = 0; i < cols; i++)
				{
					sum += data[r][i] * mat.data[i][c];
				}
				return sum;
			}
		});
	}
	
	/**
	 * This takes a #MatFunc and performs it with each single element
	 * in this matrix.
	 * <br><br><b>NOTE: This modifies the current matrix!</b>
	 * 
	 * @param func The function to perform on each element
	 * 
	 * @return this
	 */
	public Mat map(MatFunc func)
	{
		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < cols; c++)
			{
				data[r][c] = func.perform(data[r][c], r, c);
			}
		}
		return this;
	}
	
	/**
	 * Converts the 2-dimensional array to a single dimension array.
	 * So { {1, 2, 3}, {4, 5, 6} } -> toArray would be {1, 2, 3, 4, 5, 6}
	 * 
	 * @return A new double array with the 2D array flattened out
	 */
	public double[] toArray()
	{
		double[] arr = new double[rows * cols];
		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < cols; c++)
			{
				arr[c + r * cols] = data[r][c];
			}
		}
		return arr;
	}
	
	/**
	 * Retrieve a full column as a linear array
	 * 
	 * @param col The column to retrieve
	 * 
	 * @return A new double array with the specified column
	 */
	public double[] getColumn(int col)
	{
		double[] column = new double[rows];
		for(int i = 0; i < rows; i++)
		{
			column[i] = data[i][col];
		}
		return column;
	}

	/**
	 * Create a complete clone of this matrix and return it.
	 */
	public Mat clone()
	{
		return new Mat(data);
	}
	
	/**
	 * @return A single line array string of this matrix
	 */
	public String toArrayString()
	{
		return Arrays.deepToString(data);
	}
	
	public String toString()
	{		
		StringBuilder sb = new StringBuilder();
		for(int r = 0; r < rows; r++)
		{
			sb.append("[");
			for(int c = 0; c < cols; c++)
			{
				sb.append(data[r][c]);

				if(c < cols - 1) sb.append(", ");
			}
			
			sb.append(']');
			if(r < rows - 1) sb.append('\n');
		}
		return sb.toString();
	}
	
	/**
	 * Create an array with one column with the given array
	 * 
	 * @param arr The array to copy and put into the matrix
	 * 
	 * @return 	A new Mat object with one column and each row and one
	 * 			element.
	 */
	public static Mat fromArray(double[] arr)
	{
		Mat mat = new Mat(arr.length, 1);
		for(int i = 0; i < arr.length; i++)
		{
			mat.data[i][0] = arr[i];
		}
		return mat;
	}
	
	/**
	 * This is a helper interface that will allow us to perform
	 * multiple operations on each single element in the {@link Mat}.
	 */
	public interface MatFunc
	{
		/**
		 * This method takes the value at the current matrix row
		 * and index and performs some operation on it. Then returns
		 * what value should now be at that index.
		 * 
		 * @param val The value at the the current row and column
		 * @param r The current row
		 * @param c The current column
		 * 
		 * @return This should return the new value at the position
		 */
		public double perform(double val, int r, int c);
	}
	
	public static MatFunc SIGMOID = new MatFunc()
	{
		@Override
		public double perform(double val, int r, int c)
		{
			return 1 / (1 + Math.exp(-val));
		}
	};
	
	public static MatFunc SIGMOID_DERIVATIVE = new MatFunc()
	{
		@Override
		public double perform(double val, int r, int c)
		{
			return val * (1 - val);
		}
	};
	
	public static MatFunc TANH = new MatFunc()
	{
		@Override
		public double perform(double val, int r, int c)
		{
			return Math.tanh(val);
		}
	};
	
	public static MatFunc TANH_DERIVATIVE = new MatFunc()
	{
		@Override
		public double perform(double val, int r, int c)
		{
			return 1 - val * val;
		}
	};
	
	private static final long serialVersionUID = 3107367440033528127L;
}
