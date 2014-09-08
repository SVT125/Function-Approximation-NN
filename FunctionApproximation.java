// Function approximation with 1 hidden layer.
// Function variables are lowerBound, upperBound, and examples are read in from the Functions holder.
// The number of units is given by numUnits.

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.analysis.function.Sigmoid;
import java.nio.file.*;
import java.io.*;
import java.util.*;

public class FunctionApproximation {
	static List<ExampleFunctionResult> trainingExamples;
	static List<ExampleFunctionResult> testExamples;
	
	static int numUnits = 16; // the number of rectangles will be numUnits/2
	
	static BlockRealMatrix theta1;
	static BlockRealMatrix theta2;
	static BlockRealMatrix pDerivative1;
	static BlockRealMatrix pDerivative2;
	static BlockRealMatrix act2;
	static BlockRealMatrix bias1 = new BlockRealMatrix(numUnits,1);
	
	static BlockRealMatrix hypothesis = new BlockRealMatrix(1,1); // layer 4, final vector result
	static BlockRealMatrix testHypothesis = new BlockRealMatrix(1,1); // the test hypothesis when calculating the cost
	
	static double learningRate; // rate multiplied by the partial derivative. Greater values mean faster convergence but possible divergence, default .01.
	static double regularizationRate; // regularization rate, default .01.
	static int numExamples = 0; // number of examples
	static double lowerBound = 0;
	static double upperBound = 80;
	
	public static void main( String[] args ) throws Exception {
		trainingExamples = readExamples("C:/Users/James/Programming/Functions/trainingexamples/", "training");
		testExamples = readExamples("C:/Users/James/Programming/Functions/testexamples/", "test");

		theta1 = new BlockRealMatrix(numUnits,1).scalarAdd(1000d);
		updateBias();
		System.out.println( "Initialized first parameter vector." );
		theta2 = randInitialize( 1, 1, numUnits );
		updateOuterWeights();
		System.out.println( "Randomly initialized second parameter vector." );
		System.out.println("---------------------");
		
		manualUpdate();
		
		learningRate = Double.parseDouble(args[1]); 
		//regularizationRate = Double.parseDouble(args[2]);
		
		//train(Integer.parseInt(args[0]));
		
		hypothesis = forwardPropagation( testExamples.get(0));
		System.out.println("---------------------");
		printHypothesis(hypothesis);
	}
	
	// Manually set the outer weights (h), essentially defines the function to approximate.
	public static void manualUpdate() {
		double[] inputs = {0d,10d,20d,30d,40d,50d,60d,70d};
		for( int i = 0; i < theta2.getRowVector(0).getDimension(); i++ ) {
			if( (i+1) % 2 != 0 )
				theta2.setEntry(0,i,inputs[i/2]);
			else
				theta2.setEntry(0,i,-theta2.getEntry(0,i-1));
		}
	}
	
	// Runs gradient descent until the cost is less than some epsilon.
	public static void train(double epsilon) {
		int iteration = 1;
		while( calculateCost() > epsilon ) {
			gradientDescent(1);
			System.out.println( "Cost of iteration " + iteration++ + " is: " + calculateCost());
		}
	}
	
	// Calculates the cost function as the average sum of squared errors.
	public static double calculateCost() {
		double cost = 0;
		for( int i = 0; i < numExamples; i++ ) { 
			testHypothesis = forwardPropagation(trainingExamples.get(i));
			cost += Math.pow(testHypothesis.getEntry(0,0) - trainingExamples.get(i).y.getEntry(0),2);	
		}
		/* regularize theta
		cost += (regularizationRate/(2*numExamples)) * (sumSquaredMatrix(theta1) + 
			sumSquaredMatrix(theta2));
		*/
		cost = ((double)(1)/(double) (numExamples)) * cost;
		return cost;
	}
	
	// Runs numIterations iterations of batch gradient descent using the class' partial derivative terms calculated from backprop and the learning rate.
	public static void gradientDescent( int numIterations ) {
		for( int i = 0; i < numIterations; i++ ) {
			for( ExampleFunctionResult ex : trainingExamples ) 
				hypothesis = forwardPropagation( ex );
			backPropagation( trainingExamples ); // updates partial derivatives
			theta2 = theta2.subtract( pDerivative2.scalarMultiply(learningRate) );
		}
	}
	
	// Runs forward propagation, given this particular neural network.
	// Can readjust to take arguments of number of units and layers.
	public static BlockRealMatrix forwardPropagation(ExampleFunctionResult ex) {
		BlockRealMatrix z2 = theta1.multiply(convertMatrix(ex.x)).add(bias1);
		act2 = convertMatrix(sigmoid(new ArrayRealVector(z2.getColumnVector(0))));
		BlockRealMatrix z3 = theta2.multiply(act2);
		// no bias for the output unit
		return convertMatrix(new ArrayRealVector(z3.getColumnVector(0)));
	}
	
	// Runs back propagation, updates the partial derivative values for one call.
	public static void backPropagation(List<ExampleFunctionResult> examples) {
		BlockRealMatrix delta1 = new BlockRealMatrix(numUnits,1);
		BlockRealMatrix delta2 = new BlockRealMatrix(1,numUnits);
		
		for( ExampleFunctionResult ex : examples ) {
			forwardPropagation(ex);
		
			BlockRealMatrix error3 = convertMatrix(convertVector(hypothesis).subtract(ex.y));
	
			ArrayRealVector derivative2 = convertVector(act2).ebeMultiply(convertVector(new BlockRealMatrix(act2.scalarMultiply(-1d).scalarAdd(1d).getData())));
			BlockRealMatrix error2 = convertMatrix(convertVector(theta2.transpose().multiply(error3))
				.ebeMultiply(derivative2));
				
			delta2 = delta2.add(error3.multiply(act2.transpose()));
		}
		double scalarDelta = 1/(double)numExamples;
		
		pDerivative2 = new BlockRealMatrix(delta2.scalarMultiply(scalarDelta).getData());
		
		/* Regularize the partial derivatives
		RealVector thetaRow1 = theta1.getColumnVector(0), thetaRow2 = theta2.getColumnVector(0);
		pDerivative2.setColumnVector(0,new ArrayRealVector(pDerivative2.getColumnVector(0)).add(thetaRow2));
		*/
	}
	
	// Sigmoid function that takes an ArrayRealVector.
	public static ArrayRealVector sigmoid( ArrayRealVector arg ) {
		return arg.mapToSelf( new Sigmoid(1e-323,1-(1e-323)) );
	}
	
	// Updates the outer weights (in this case theta2) so that each pair of activation units have outgoing weights +h and -h respectively.
	// Takes the first unit to be +h and updates the second unit of the pair to be -h.
	public static void updateOuterWeights() {
		double h = 0;
		for( int i = 0; i < theta2.getColumnVector(0).getDimension(); i++ ) {
			if( (i+1) % 2 != 0 )
				h = theta2.getEntry(i,0);
			else
				theta2.setEntry(i,0,-h);
		}
	}
	
	// Initialize the first weight according to the steps; in particular, we adjust the bias which is dependent on the weight such that b = -ws.
	public static void updateBias() {
		double stepIncrement = (upperBound - lowerBound)/numUnits;
		int index = 0;
		for( double i = 0; i < upperBound; i += stepIncrement ) {
			double bias = (-theta1.getEntry(index,0)) * i;
			bias1.setEntry(index,0,bias);
			index++;
		}
	}
	
	// Randomly initialize each of the theta values by [negEpsilon, epsilon] or [-epsilon, epsilon].
	// Can rework the method to initialize more optimally, current naive implementation in quadratic time.
	public static BlockRealMatrix randInitialize( double epsilon, int row, int col ) {
		BlockRealMatrix mat = new BlockRealMatrix(row,col);
		Random r = new Random();
		for( int i = 0; i < row; i++ )
			for( int j = 0; j < col; j++ ) {
				double rand = r.nextDouble() * (2*epsilon) - epsilon;
				mat.addToEntry(i,j,rand);
			}
		return mat;
	}
		
	// Read in the examples' input and output by reading their RGB values and the name (per format), stored in a List.
	public static List<ExampleFunctionResult> readExamples(String path, String set ) {
		int loadedExamples = 0;
		List<ExampleFunctionResult> ex = new ArrayList<ExampleFunctionResult>();
		Path dir = Paths.get(path);
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				BufferedReader br = new BufferedReader(new FileReader(file.toString()));
				String line;
				while ((line = br.readLine()) != null) {
					String[] example = line.split(" ");
					ex.add( new ExampleFunctionResult(Double.parseDouble(example[0]),Double.parseDouble(example[1])) );
					loadedExamples++;
				}
				br.close();	
			}
		} catch (IOException | DirectoryIteratorException x) {
			System.err.println(x);
		}
		if( set.equals( "training" ) )
			numExamples = loadedExamples;
		
		System.out.println( "Loaded in " + loadedExamples + " " + set + " examples." );
		return ex;
	}
	
	// Sums the matrix whose elements have been squared.
	public static double sumSquaredMatrix( BlockRealMatrix mat ) {
		double sum = 0;
		for( int i = 0; i < mat.getRowDimension(); i++ ) {
			for( int j = 0; j < mat.getColumnDimension(); j++ ) {
				sum += Math.pow(mat.getEntry(i,j),2);
			}
		}
		return sum;
	}
	
	// Sums the given vector, returns a double value.
	public static double sumVector( ArrayRealVector vec ) {
		double sum = 0;
		for( int i = 0; i < vec.getDimension(); i++ )
			sum += vec.getEntry(i);
		return sum;
	}
	
	// Print the hypothesis - take the highest value across all the entries and convert to the character.
	// Assumes hypothesis is a 1 x n matrix.
	public static void printHypothesis( BlockRealMatrix hyp ) {
		System.out.println( "The hypothesized output is: " + hyp.getEntry(0,0));
	}
	
	// Auxiliary method to turn an ArrayRealVector to a BlockRealMatrix.
	public static BlockRealMatrix convertMatrix( ArrayRealVector vec ) {
		BlockRealMatrix mat = new BlockRealMatrix(vec.getDimension(),1);
		for( int i = 0; i < vec.getDimension(); i++ )
			mat.setEntry( i, 0, vec.getEntry(i));
		return mat;
	}

	// Auxiliary method to turn a BlockRealMatrix to an ArrayRealVector - works only if one dimension of the matrix = 1.
	public static ArrayRealVector convertVector( BlockRealMatrix mat ) {
		if( mat.getRowVector(0).getDimension() == 1 )
			return new ArrayRealVector(mat.getColumnVector(0));
		else
			return new ArrayRealVector(mat.getRowVector(0));
	}
		
	// Auxiliary method to check the dimensions of the input matrix.
	public static void printDimensions( BlockRealMatrix mat ) {
		System.out.println( "Number of rows: " + mat.getColumnVector(0).getDimension() );
		System.out.println( "Number of columns: " + mat.getRowVector(0).getDimension() );		
	}
	
	// Auxiliary method to better examine the matrix's values.
	public static void printMatrixToFile( BlockRealMatrix mat ) throws Exception {
		PrintWriter writer = new PrintWriter("matrixdump.txt", "UTF-8");
		double[][] matArray = mat.getData();
		for( int i = 0; i < matArray.length; i++ ) {
			for( int j = 0; j < matArray[i].length; j++ ) {
				writer.print( matArray[i][j] + " " );
			}
			writer.println();
		}
	}
	
	// Auxiliary method to check proper vector accesses.
	// Prints the matrix by creating an 2d double array primitive to loop over, works for non n x n matrices.
	public static void printMatrix( BlockRealMatrix brm ) {
		double[][] mat = brm.getData();
		for( int i = 0; i < mat.length; i++ ) {
			for( int j = 0; j < mat[i].length; j++ ) {
				System.out.print( mat[i][j] + " " );
			}
			System.out.println();
		}
	}
	
	// Auxiliary method to print the vector.
	public static void printVector( ArrayRealVector vec ) {
		double[] array = vec.toArray();
		for( int i = 0; i < array.length; i++ ) {
			System.out.println( array[i] );
		}
	}
}