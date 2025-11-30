#include <bits/stdc++.h>
using namespace std;
#define FIO ios::sync_with_stdio(false), cin.tie(nullptr), cout.tie(nullptr)
#define ll long long


void bfs(ll node , vector<vector<ll>>& graph , vector<ll>& visited){
    
    if(visited[node]) return ;
    
    visited[node] = 1 ;
    
    queue<ll> q ;
    q.push(node) ;
    
    while(!q.empty()) {
        ll top = q.front() ;
        q.pop() ;
        cout << top << " " ;
        for(auto child : graph[top]) {
            q.push(child) ;
        }
    }
}
void solve() {
    ll n , m;
    cin >> n >> m;
    vector<vector<ll>> graph(n, vector<ll>()) ;
    vector<ll> visited(n) ;
    
    while(m--) {
        ll x ,y ;
        cin >> x >> y ;
        graph[x].push_back(y) ; // ->
    }
    
    bfs(0,graph,visited) ;
}

int main() {
    FIO;
    int t = 1;
    // cin >> t;
    while (t--) solve();
    return 0;
}