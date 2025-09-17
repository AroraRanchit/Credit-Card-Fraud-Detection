import os
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Detect protocol automatically based on MAIL_PORT
mail_port = int(os.getenv("MAIL_PORT", 587))

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_SERVER=os.getenv("MAIL_SERVER"),
    MAIL_PORT=mail_port,
    MAIL_STARTTLS=(mail_port == 587),   # Use STARTTLS if port is 587
    MAIL_SSL_TLS=(mail_port == 465)     # Use SSL if port is 465
)


async def send_email(subject: str, recipients: list, body: str):
    """
    Send an email with subject, recipients, and body.
    """
    message = MessageSchema(
        subject=subject,
        recipients=recipients,  # List of email addresses
        body=body,
        subtype="html"          # Send HTML emails (for analytics-style content)
    )
    
    fm = FastMail(conf)
    await fm.send_message(message)
    print(f"✅ Email sent to {recipients}")


# Example: sending analytics email
if __name__ == "__main__":
    import asyncio
    example_html = """
    <h2>📊 Credit Card Fraud Detection Report</h2>
    <p>Dear User,</p>
    <p>Here are your transaction analytics:</p>
    <ul>
        <li>Total Transactions: 150</li>
        <li>Fraudulent Transactions: 3</li>
        <li>Accuracy of Model: 96%</li>
    </ul>
    <p>Stay safe,<br/>Your Fraud Detection System</p>
    """
    asyncio.run(send_email(
        subject="Your Credit Card Fraud Analytics Report",
        recipients=["test@example.com"],  # Change this
        body=example_html
    ))
