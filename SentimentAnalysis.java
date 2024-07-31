import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class SentimentAnalysis extends Configured implements Tool {

    public static class SentimentMapper extends Mapper<Object, Text, Text, IntWritable> {

        private static final String API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis";
        private static final String AUTH_TOKEN = "Bearer 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";

        private static final IntWritable ONE = new IntWritable(1);

        private static int analyzeSentiment(String document) throws IOException {
            URL url = new URL(API_URL);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Authorization", AUTH_TOKEN);
            connection.setRequestProperty("Content-Type", "application/json; utf-8");
            connection.setDoOutput(true);

            String payload = "{\"text\": \"" + document + "\"}";
            try (OutputStream os = connection.getOutputStream()) {
                byte[] input = payload.getBytes(StandardCharsets.UTF_8);
                os.write(input, 0, input.length);
            }

            int responseCode = connection.getResponseCode();
            if (responseCode != 200) {
                throw new IOException("Failed : HTTP error code : " + responseCode);
            }

            try (BufferedReader br = new BufferedReader(new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8))) {
                StringBuilder response = new StringBuilder();
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }

                JSONArray jsonArray = new JSONArray(response.toString());
                JSONObject jsonObject = jsonArray.getJSONObject(0);
                String label = jsonObject.getString("label");

                switch (label) {
                    case "POS":
                        return 1;
                    case "NEG":
                        return -1;
                    default:
                        return 0;
                }
            }
        }

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String document = value.toString().trim();
            int sentimentScore = analyzeSentiment(document);

            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String fileName = fileSplit.getPath().getName();
            Text outKey = new Text(fileName);

            if (sentimentScore > 0) {
                context.write(new Text(outKey.toString() + "\tpositive"), ONE);
            } else if (sentimentScore < 0) {
                context.write(new Text(outKey.toString() + "\tnegative"), ONE);
            } else {
                context.write(new Text(outKey.toString() + "\tneutral"), ONE);
            }
        }
    }

    public static class SentimentCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    public static class SentimentReducer extends Reducer<Text, IntWritable, Text, Text> {

        private Map<String, Integer> positiveCounts = new HashMap<>();
        private Map<String, Integer> negativeCounts = new HashMap<>();
        private Map<String, Integer> neutralCounts = new HashMap<>();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            String[] keyParts = key.toString().split("\t");
            String fileName = keyParts[0];
            String sentiment = keyParts[1];

            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }

            switch (sentiment) {
                case "positive":
                    positiveCounts.put(fileName, positiveCounts.getOrDefault(fileName, 0) + sum);
                    break;
                case "negative":
                    negativeCounts.put(fileName, negativeCounts.getOrDefault(fileName, 0) + sum);
                    break;
                case "neutral":
                    neutralCounts.put(fileName, neutralCounts.getOrDefault(fileName, 0) + sum);
                    break;
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (String fileName : positiveCounts.keySet()) {
                int positive = positiveCounts.getOrDefault(fileName, 0);
                int negative = negativeCounts.getOrDefault(fileName, 0);
                int neutral = neutralCounts.getOrDefault(fileName, 0);

                String maxLabel;
                if (positive > negative && positive > neutral/2) {
                    maxLabel = "positive";
                } else if (negative > positive && negative > neutral/2) {
                    maxLabel = "negative";
                } else {
                    maxLabel = "neutral";
                }
                context.write(new Text(fileName), new Text(maxLabel));
            }
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: SentimentAnalysis <input path> <output path>");
            return -1;
        }

        Configuration conf = getConf();
        Job job = Job.getInstance(conf, "Sentiment Analysis");

        job.setJarByClass(SentimentAnalysis.class);
        job.setMapperClass(SentimentMapper.class);
        job.setCombinerClass(SentimentCombiner.class);
        job.setReducerClass(SentimentReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new SentimentAnalysis(), args);
        System.exit(res);
    }
}
