from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
import io

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read the contents of the file
        contents = await file.read()
        
        # Create a StringIO object from the contents
        from io import StringIO
        import io
        
        # Try to decode the contents as UTF-8
        try:
            str_io = StringIO(contents.decode('utf-8'))
        except UnicodeDecodeError:
            # If UTF-8 fails, try to read as bytes with pandas
            str_io = io.BytesIO(contents)
        
        # Read CSV file
        try:
            df = pd.read_csv(str_io)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'datetime']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            if df['timestamp'].max() > 1e10:  # Convert from milliseconds to seconds if needed
                df['timestamp'] = df['timestamp'] / 1000
        
        # Add datetime column if not present
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Return processed data
        return {
            "data": df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
