from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import httpx
import os

CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL", "")

security = HTTPBearer()

def get_jwks():
    if not CLERK_JWKS_URL:
        return None
    try:
        response = httpx.get(CLERK_JWKS_URL)
        return response.json()
    except Exception as e:
        print(f"Failed to fetch JWKS: {e}")
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    token = credentials.credentials
    if not CLERK_JWKS_URL:
        # If Clerk isn't configured, fallback to a dummy user (e.g. local testing before keys are set)
        print("[AUTH] Clerk JWKS URL not configured! Using fallback user.")
        return "fallback_user"
        
    try:
        jwks = get_jwks()
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
        
        if rsa_key:
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(rsa_key)
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                # Clerk tokens require checking the party/issuer depending on strictness
                options={"verify_aud": False, "verify_iss": False},
                leeway=60
            )
            return payload["sub"] # The Clerk user ID
        else:
            raise HTTPException(status_code=401, detail="Invalid token key")
            
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication token: {e}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {e}")
