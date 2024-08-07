AWSTemplateFormatVersion: '2010-09-09'
Resources:
  GlueJob:
    Type: AWS::Glue::Job
    Properties:
      Name: "glue-job-ingestion"
      Role: "arn:aws:iam::233359079168:role/LabRole"  # Replace with the ARN of the 'Lab Role'
      Command:
        Name: "glueetl"
        ScriptLocation: "s3://datasource-dataops-group5/script-ingestion/ingestion.py"  # Update with your S3 path
      DefaultArguments:
        "--job-language": "python"
      MaxRetries: 0
      GlueVersion: "2.0"
      WorkerType: "G.1X"
      NumberOfWorkers: 5
