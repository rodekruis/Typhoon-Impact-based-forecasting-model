import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


def sendemail(smtp_server: str,
              smtp_port: int,
              email_username: str,
              email_password: str,
              email_subject: str,
              from_address: str,
              to_address_list: list,
              cc_address_list: list,
              message_html: str,
              csv_filename: str,
              image_filename: str):

    # Create message
    message = MIMEMultipart("alternative")
    message["Subject"] = email_subject
    message["From"] = from_address
    message["To"] = ", ".join(to_address_list)
    message["CC"] = ", ".join(cc_address_list)
    message.attach(MIMEText(message_html, "html"))

    # Attach image and csv
    for filename in [csv_filename, image_filename]:
        with open(filename, 'rb') as file:
            # Attach the file with filename to the email
            message.attach(MIMEApplication(file.read(), Name=filename))

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls(context=context)
        server.login(email_username, email_password)
        server.sendmail(from_address,
                        to_address_list + cc_address_list,
                        message.as_string())


def sendemail(from_addr, to_addr_list, cc_addr_list, message, login, password, smtpserver):
    header  = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list)
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    return problems


def sendemail_gmail(from_addr, to_addr_list, cc_addr_list,login,password, message,smtpserver='smtp.gmail.com:587'):
    header  = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list) 
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)# message)
    server.quit()
    return problems


def create_message_with_attachment(sender, to, subject, message_text, file):
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
  message['to'] = to
  message['from'] = sender
  message['subject'] = subject

  msg = MIMEText(message_text)
  message.attach(msg)

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
  message.attach(msg)
  #return {'raw': base64.urlsafe_b64encode(message.as_string())}
  return msg# {'raw': base64.urlsafe_b64encode(str.encode(message.as_string()))}
