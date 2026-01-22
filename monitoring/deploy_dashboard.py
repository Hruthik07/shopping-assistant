"""Deploy CloudWatch dashboard via boto3."""
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from monitoring.cloudwatch_dashboard import dashboard_generator
from src.analytics.logger import logger
from src.utils.config import settings


def deploy_dashboard(dashboard_name: str = "ShoppingAssistant-Dashboard"):
    """Deploy CloudWatch dashboard.
    
    Args:
        dashboard_name: Name of the dashboard
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        region = getattr(settings, 'aws_region', 'us-east-1')
        client = boto3.client('cloudwatch', region_name=region)
        
        # Generate dashboard JSON
        dashboard_body = dashboard_generator.generate_dashboard()
        
        # Deploy dashboard
        try:
            response = client.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            logger.info(f"Dashboard '{dashboard_name}' deployed successfully")
            logger.info(f"View at: https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#dashboards:name={dashboard_name}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'AccessDenied':
                logger.error("Access denied. Check IAM permissions for cloudwatch:PutDashboard")
            else:
                logger.error(f"Failed to deploy dashboard: {e}")
            return False
        except NoCredentialsError:
            logger.error("AWS credentials not found. Set up AWS credentials or IAM role.")
            return False
            
    except ImportError:
        logger.error("boto3 not installed. Install with: pip install boto3")
        return False
    except Exception as e:
        logger.error(f"Unexpected error deploying dashboard: {e}")
        return False


if __name__ == "__main__":
    dashboard_name = sys.argv[1] if len(sys.argv) > 1 else "ShoppingAssistant-Dashboard"
    success = deploy_dashboard(dashboard_name)
    sys.exit(0 if success else 1)
