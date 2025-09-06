import pandas as pd
import io
import requests
import json
from datetime import datetime

class EmailAssistant:
    """
    An AI-powered assistant for processing and managing support emails.
    This class handles prioritization, categorization, and response generation.
    """

    def __init__(self, file_path):

        try:
            # pandas can read directly from a URL that links to raw file content.
            # This works for both local file paths and remote URLs.
            self.df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data from source: {e}")
            self.df = pd.DataFrame()

    def _get_priority(self, subject, body):
        """
        Assigns a priority level (High, Medium, Low) based on keywords.
        
        Args:
            subject (str): The email subject.
            body (str): The email body.
            
        Returns:
            str: The determined priority level.
        """
        high_priority_keywords = ['urgent', 'critical', 'immediate', 'blocked', 'downtime', 'error']
        medium_priority_keywords = ['help', 'support', 'issue', 'problem', 'question', 'query']

        text = (subject + ' ' + body).lower()
        if any(keyword in text for keyword in high_priority_keywords):
            return 'High'
        if any(keyword in text for keyword in medium_priority_keywords):
            return 'Medium'
        return 'Low'

    def _categorize_email(self, subject, body):
        """
        Categorizes an email based on its content.
        
        Args:
            subject (str): The email subject.
            body (str): The email body.
            
        Returns:
            str: The determined category.
        """
        text = (subject + ' ' + body).lower()
        if 'login' in text or 'access' in text or 'password' in text or 'account' in text:
            return 'Login/Account'
        if 'billing' in text or 'pricing' in text or 'subscription' in text or 'refund' in text:
            return 'Billing/Pricing'
        if 'api' in text or 'integration' in text:
            return 'Integration'
        if 'downtime' in text or 'servers' in text:
            return 'System Downtime'
        return 'General Inquiry'

    def _get_ai_response(self, prompt, context):
        """
        Simulates an API call to a Gemini model to generate a response.
        In a real application, you would replace this with actual API credentials.
        
        Args:
            prompt (str): The prompt for the AI model.
            context (str): The email body or other relevant context.
            
        Returns:
            str: The generated response.
        """
        try:
            # Construct the payload for the API call
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"text": context}
                    ]
                }],
                "tools": [{"google_search": {}}],  # Use search grounding for up-to-date info
                "systemInstruction": {
                    "parts": [{"text": "You are a friendly and helpful customer support representative. Generate a concise, professional response to the customer's email."}]
                },
            }

            # API endpoint and key are handled by the system.
            api_key = ""
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

            # Make the API call
            response = requests.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raise an exception for bad status codes
            
            # Parse the response to get the generated text
            result = response.json()
            generated_text = result['candidates'][0]['content']['parts'][0]['text']
            
            return generated_text
        except Exception as e:
            # Fallback to a predefined response if the API call fails
            print(f"API call failed: {e}. Using a fallback response.")
            return "Thank you for reaching out. We have received your request and will get back to you shortly."


    def process_emails(self):
        """
        Processes each email in the dataset to prioritize, categorize, and
        generate a response.
        
        Returns:
            pd.DataFrame: The DataFrame with new columns for priority, category, and response.
        """
        if self.df.empty:
            print("No data to process. Exiting.")
            return self.df

        self.df['priority'] = self.df.apply(
            lambda row: self._get_priority(row['subject'], row['body']),
            axis=1
        )
        self.df['category'] = self.df.apply(
            lambda row: self._categorize_email(row['subject'], row['body']),
            axis=1
        )
        self.df['response'] = self.df.apply(
            lambda row: self._get_ai_response(
                f"Draft a response to the following email from {row['sender']} regarding '{row['subject']}':",
                row['body']
            ),
            axis=1
        )
        
        return self.df


def main():
    """
    Main function to run the email assistant.
    """
    # Replace this placeholder URL with the raw file URL from your GitHub repository.
    # The format is: https://raw.githubusercontent.com/{username}/{repository_name}/{branch_name}/{file_path}
    file_path = "https://github.com/mparjun2k1/Unstop-AI-fresher-challenge/blob/main/68b1acd44f393_Sample_Support_Emails_Dataset.csv"
    
    print(f"Initializing AI Email Assistant with file from: {file_path}...")
    assistant = EmailAssistant(file_path)
    
    print("Processing emails...")
    processed_df = assistant.process_emails()
    
    # Sort the DataFrame by priority (High, Medium, Low) for dashboard display
    priority_order = pd.Categorical(processed_df['priority'], categories=['High', 'Medium', 'Low'], ordered=True)
    processed_df['priority'] = priority_order
    processed_df.sort_values(by='priority', inplace=True)
    
    print("\n--- Processed Emails Dashboard ---")
    print(processed_df[['sender', 'subject', 'priority', 'category', 'response']].to_markdown(index=False, numalign="left", stralign="left"))
    
if __name__ == "__main__":
    main()
