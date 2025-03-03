import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class GmailSender:
    def __init__(self, sender_email, sender_password):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587  # Standard port for TLS

    def send_email(self, recipient_email, subject, body):
        try:
            # Create the email message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            # Connect to Gmail's SMTP server
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()  # Secure the connection using TLS
            server.login(self.sender_email, self.sender_password)

            # Send the email
            server.sendmail(self.sender_email, recipient_email, msg.as_string())
            print(f"Email successfully sent to {recipient_email}")

            # Close the connection
            server.quit()

        except Exception as e:
            print(f"Failed to send email. Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    sender = GmailSender('your-email@gmail.com', 'your-password')
    sender.send_email('recipient-email@gmail.com', 'Test Subject', 'This is the email body.')
