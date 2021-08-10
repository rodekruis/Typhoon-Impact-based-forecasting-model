import base64
import email
import os
from os import listdir
from os.path import isfile, join 
from os.path import basename
from email.mime.application import MIMEApplication 
from email.utils import COMMASPACE, formatdate
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
import smtplib, ssl



 
def file_attach_2message(file):
  """Create a message for an email.

  Args:
     file: The path to the file to be attached.

  Returns:
    An object containing email object.
  """

  content_type, encoding = mimetypes.guess_type(file)
  if content_type is None or encoding is not None:
    content_type = 'application/octet-stream'
  main_type, sub_type = content_type.split('/', 1)
  if main_type == 'text':
    fp = open(file, 'rb')
    msg = MIMEText(fp.read(), _subtype=sub_type)
    fp.close()
  elif main_type == 'image':
    fp = open(file, 'rb')
    msg = MIMEImage(fp.read(), _subtype=sub_type)
    fp.close()
  elif main_type == 'csv':
    fp = open(file, 'rb')
    msg = MIMEApplication(fp.read(), _subtype=sub_type)
    fp.close()
  elif main_type == 'audio':
    fp = open(file, 'rb')
    msg = MIMEAudio(fp.read(), _subtype=sub_type)
    fp.close()
  else:
    fp = open(file, 'rb')
    msg = MIMEBase(main_type, sub_type)
    msg.set_payload(fp.read())
    fp.close()
  filename = os.path.basename(file)
  msg.add_header('Content-Disposition', 'attachment', filename=filename)
  return msg
  
def create_message_with_attachment(message_text,file,file2,file1):
  """Create a message for an email.

  Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.
    file: The path to the file to be attached.

  Returns:
    An object containing a base64url encoded email object.
  """
  message = MIMEMultipart()
  msg = MIMEText(message_text)
  message.attach(msg)  
  msg=file_attach_2message(file)
  message.attach(msg)
  msg=file_attach_2message(file1)
  message.attach(msg)
  msg=file_attach_2message(file2)
  message.attach(msg)
  return message#.as_string()# {'raw': base64.urlsafe_b64encode(str.encode(message.as_string()))}

 
def sendemail(emails, subject, content,smtp_server_domain_name,port,sender_mail,password):
    ssl_context = ssl.create_default_context()
    service = smtplib.SMTP_SSL(smtp_server_domain_name, port, context=ssl_context)
    service.login(sender_mail, password)
    
    for email in emails:
        result = service.sendmail(sender_mail, email, f"Subject: {subject}\n{content}")

    service.quit()
 
    

